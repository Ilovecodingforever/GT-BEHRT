from __future__ import annotations

import itertools
import pickle
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

EDGE_TYPES = {
    ("diag", "diag"): 0,
    ("med", "med"): 1,
    ("proc", "proc"): 2,
    ("diag", "med"): 3,
    ("diag", "proc"): 4,
    ("med", "proc"): 5,
    ("vst", "code"): 6,
}


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _pair(a: str, b: str) -> tuple[str, str]:
    if "vst" in (a, b):
        return ("vst", "code")
    return tuple(sorted((a, b)))


def trim_typed_codes(typed_codes, max_codes_per_visit: int | None):
    if max_codes_per_visit is None:
        return typed_codes
    max_codes_per_visit = int(max_codes_per_visit)
    if max_codes_per_visit <= 0 or len(typed_codes) <= max_codes_per_visit:
        return typed_codes
    return typed_codes[:max_codes_per_visit]


def truncate_materialized_graph(graph, max_codes_per_visit: int | None):
    if max_codes_per_visit is None:
        return graph
    max_codes_per_visit = int(max_codes_per_visit)
    if max_codes_per_visit <= 0:
        return graph
    max_nodes = max_codes_per_visit + 1
    if int(graph.x.shape[0]) <= max_nodes:
        return graph
    graph = graph.clone() if hasattr(graph, "clone") else graph
    keep = graph.edge_index[0].lt(max_nodes) & graph.edge_index[1].lt(max_nodes)
    graph.x = graph.x[:max_nodes]
    graph.edge_index = graph.edge_index[:, keep]
    graph.edge_attr = graph.edge_attr[keep]
    return graph


def make_visit_graph(visit, label: int, active: bool = True, max_codes_per_visit: int | None = None):
    from torch_geometric.data import Data
    if not active:
        return Data(
            x=torch.tensor([0], dtype=torch.long),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.tensor([EDGE_TYPES[("vst", "code")]], dtype=torch.long),
            age=torch.tensor([0], dtype=torch.long),
            time=torch.tensor([0], dtype=torch.long),
            delta=torch.tensor([0], dtype=torch.long),
            adm_type=torch.tensor([0], dtype=torch.long),
            los=torch.tensor([0], dtype=torch.long),
            mask_v=torch.tensor([0], dtype=torch.long),
            label=torch.tensor([int(label)], dtype=torch.float),
            mask=torch.tensor([1], dtype=torch.long),
        )
    delta, typed_codes, meta = int(visit[0][0]), trim_typed_codes(visit[1], max_codes_per_visit), visit[2]
    x = [0] + [int(code_id) + 1 for code_id, _ in typed_codes]
    node_types = ["vst"] + [code_type for _, code_type in typed_codes]
    edges = []
    edge_attrs = []
    for src, dst in itertools.permutations(range(len(x)), 2):
        edges.append([src, dst])
        edge_attrs.append(EDGE_TYPES[_pair(node_types[src], node_types[dst])])
    if not edges:
        edges = [[0, 0]]
        edge_attrs = [EDGE_TYPES[("vst", "code")]]
    return Data(
        x=torch.tensor(x, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.long),
        age=torch.tensor([_clamp(meta.get("age", 0), 0, 102)], dtype=torch.long),
        time=torch.tensor([_clamp(meta.get("day_of_year", 0), 0, 366)], dtype=torch.long),
        delta=torch.tensor([_clamp(delta, 0, 143)], dtype=torch.long),
        adm_type=torch.tensor([_clamp(meta.get("adm_type", 0), 0, 10)], dtype=torch.long),
        los=torch.tensor([_clamp(meta.get("los", 0), 0, 1191)], dtype=torch.long),
        mask_v=torch.tensor([1], dtype=torch.long),
        label=torch.tensor([int(label)], dtype=torch.float),
        mask=torch.tensor([1], dtype=torch.long),
    )


def load_gtbehrt_pid_to_index(combined_path: str | Path) -> dict[int, int]:
    combined_path = Path(combined_path)
    if combined_path.is_dir():
        with (combined_path / "index.pkl").open("rb") as handle:
            index = pickle.load(handle)
        return {int(pid): int(idx) for pid, idx in index["pid_to_index"].items()}
    with combined_path.open("rb") as handle:
        combined = pickle.load(handle)
    return {int(patient[0]): idx for idx, patient in enumerate(combined)}


class _ShardedDatasetBase(Sequence[list]):
    def __init__(self, root: str | Path, max_seq_len: int = 50, max_codes_per_visit: int | None = None):
        self.root = Path(root)
        self.max_seq_len = int(max_seq_len)
        self.max_codes_per_visit = None if max_codes_per_visit is None else int(max_codes_per_visit)
        with (self.root / "index.pkl").open("rb") as handle:
            self.index = pickle.load(handle)
        self.shards = list(self.index.get("shards", []))
        self.length = int(self.index.get("length", 0))
        self.cumulative_counts = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative_counts.append(total)
        self._cached_shard_name = None
        self._cached_records = None

    def __len__(self) -> int:
        return self.length

    def _locate(self, index: int) -> tuple[dict, int]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_idx = bisect_right(self.cumulative_counts, index)
        prior = 0 if shard_idx == 0 else self.cumulative_counts[shard_idx - 1]
        return self.shards[shard_idx], index - prior

    def _load_shard(self, shard: dict):
        shard_name = shard["file"]
        if self._cached_shard_name != shard_name:
            with (self.root / shard_name).open("rb") as handle:
                self._cached_records = pickle.load(handle)
            self._cached_shard_name = shard_name
        return self._cached_records


class GTBEHRTLazyDataset(Sequence[list]):
    def __init__(self, combined: list, max_seq_len: int = 50, max_codes_per_visit: int | None = None):
        self.combined = combined
        self.max_seq_len = int(max_seq_len)
        self.max_codes_per_visit = None if max_codes_per_visit is None else int(max_codes_per_visit)

    @classmethod
    def from_combined_path(cls, path: str | Path, max_seq_len: int = 50, max_codes_per_visit: int | None = None) -> "GTBEHRTLazyDataset":
        with Path(path).open("rb") as handle:
            combined = pickle.load(handle)
        return cls(combined, max_seq_len=max_seq_len, max_codes_per_visit=max_codes_per_visit)

    def __len__(self) -> int:
        return len(self.combined)

    def __getitem__(self, index: int) -> list:
        subject_id, label_obj, visits = self.combined[index]
        del subject_id
        label = int(label_obj[0])
        graphs = [make_visit_graph(visit, label, active=True, max_codes_per_visit=self.max_codes_per_visit) for visit in visits[: self.max_seq_len]]
        while len(graphs) < self.max_seq_len:
            graphs.append(make_visit_graph(None, label, active=False))
        return graphs


class GTBEHRTShardedCombinedDataset(_ShardedDatasetBase):
    def __getitem__(self, index: int) -> list:
        shard, offset = self._locate(index)
        patient = self._load_shard(shard)[offset]
        label = int(patient[1][0])
        graphs = [make_visit_graph(visit, label, active=True, max_codes_per_visit=self.max_codes_per_visit) for visit in patient[2][: self.max_seq_len]]
        while len(graphs) < self.max_seq_len:
            graphs.append(make_visit_graph(None, label, active=False))
        return graphs


class GTBEHRTMaterializedDataset(Sequence[list]):
    def __init__(self, graphs_by_patient: list[list], max_seq_len: int = 50, max_codes_per_visit: int | None = None):
        self.graphs_by_patient = graphs_by_patient
        self.max_seq_len = int(max_seq_len)
        self.max_codes_per_visit = None if max_codes_per_visit is None else int(max_codes_per_visit)

    @classmethod
    def from_data_path(cls, path: str | Path, max_seq_len: int = 50, max_codes_per_visit: int | None = None) -> "GTBEHRTMaterializedDataset":
        with Path(path).open("rb") as handle:
            graphs_by_patient = pickle.load(handle)
        return cls(graphs_by_patient, max_seq_len=max_seq_len, max_codes_per_visit=max_codes_per_visit)

    def __len__(self) -> int:
        return len(self.graphs_by_patient)

    def __getitem__(self, index: int) -> list:
        graphs = [truncate_materialized_graph(graph, self.max_codes_per_visit) for graph in self.graphs_by_patient[index][: self.max_seq_len]]
        label = 0
        if graphs:
            first_label = getattr(graphs[0], "label", None)
            if first_label is not None and len(first_label) > 0:
                label = int(first_label[0].item())
        while len(graphs) < self.max_seq_len:
            graphs.append(make_visit_graph(None, label, active=False))
        return graphs


class GTBEHRTShardedMaterializedDataset(_ShardedDatasetBase):
    def __getitem__(self, index: int) -> list:
        shard, offset = self._locate(index)
        graphs = [truncate_materialized_graph(graph, self.max_codes_per_visit) for graph in self._load_shard(shard)[offset][: self.max_seq_len]]
        label = 0
        if graphs:
            first_label = getattr(graphs[0], "label", None)
            if first_label is not None and len(first_label) > 0:
                label = int(first_label[0].item())
        while len(graphs) < self.max_seq_len:
            graphs.append(make_visit_graph(None, label, active=False))
        return graphs


class GTBEHRTIndexSubset(Dataset):
    def __init__(self, dataset: Sequence[list], indices: Iterable[int]):
        self.dataset = dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> list:
        return self.dataset[self.indices[index]]


def load_gtbehrt_dataset(data_path: str | Path, max_seq_len: int = 50, max_codes_per_visit: int | None = None):
    data_path = Path(data_path)
    if data_path.is_dir() and (data_path / "index.pkl").exists():
        return GTBEHRTShardedMaterializedDataset(data_path, max_seq_len=max_seq_len, max_codes_per_visit=max_codes_per_visit), "materialized"
    if data_path.exists():
        return GTBEHRTMaterializedDataset.from_data_path(data_path, max_seq_len=max_seq_len, max_codes_per_visit=max_codes_per_visit), "materialized"
    combined_path = data_path.parent / "pipeline_output.combined.train"
    if combined_path.is_dir() and (combined_path / "index.pkl").exists():
        return GTBEHRTShardedCombinedDataset(combined_path, max_seq_len=max_seq_len, max_codes_per_visit=max_codes_per_visit), "lazy"
    if combined_path.exists():
        return GTBEHRTLazyDataset.from_combined_path(combined_path, max_seq_len=max_seq_len, max_codes_per_visit=max_codes_per_visit), "lazy"
    raise FileNotFoundError(f"Neither {data_path} nor {combined_path} exists")
