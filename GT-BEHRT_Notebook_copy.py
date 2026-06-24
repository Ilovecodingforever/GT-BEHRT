# Auto-generated from GT-BEHRT_Notebook.ipynb

# %% [cell 0]
# !pip install import_ipynb
# !pip install -U -q PyDrive
# !pip install pytorch_pretrained_bert
# !pip install sparse
# !pip install transformers
# !pip install torchmetrics
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# %% [cell 1]
# !pip install einops

# %% [cell 2]
# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.

import torch
from torch_geometric.data import Data

import numpy as np
import sparse

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgmnn
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataListLoader as GraphLoader
from torch_geometric.data import Batch

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import time
from sklearn import preprocessing
import math
from torch.utils.data import Dataset
import copy
import sklearn.metrics as skm
import pandas as pd
import random
from torch.utils.data.dataset import Dataset
import pytorch_pretrained_bert as Bert
import itertools
from einops import rearrange, repeat

# %% [cell 3]
import ast
from typing import Optional, Tuple, Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import LayerNorm
import torch.nn.functional as F
from torch import Tensor

class TransformerConv(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.layernorm1 = LayerNorm(out_channels)
        self.layernorm2 = LayerNorm(out_channels)
        self.gelu = nn.GELU()
        self.proj = Linear(heads * out_channels, out_channels)
        self.ffn = Linear(out_channels, out_channels)
        self.ffn2 = Linear(out_channels, out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, batch=None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels
        residual = x
        x = self.layernorm1(x, batch)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = self.proj(out.view(-1, self.heads * self.out_channels))
        else:
            out = out.mean(dim=1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out+residual
        residual = out

        out = self.layernorm2(out)
        out = self.gelu(self.ffn(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.ffn2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out + residual
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:


        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GraphTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv = tgmnn.Sequential('x, edge_index, edge_attr, batch', [
            (TransformerConv(config.hidden_size // 5, config.hidden_size // 5, heads=2, edge_dim=config.hidden_size // 5, dropout=config.hidden_dropout_prob, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (TransformerConv(config.hidden_size // 5, config.hidden_size // 5, heads=2, edge_dim=config.hidden_size // 5, dropout=config.hidden_dropout_prob, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (TransformerConv(config.hidden_size // 5, config.hidden_size // 5, heads=2, edge_dim=config.hidden_size // 5, dropout=config.hidden_dropout_prob, concat=False), 'x, edge_index, edge_attr -> x'),
        ])

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size // 5)
        self.embed_ee = nn.Embedding(7, config.hidden_size // 5)

    def forward(self, x, edge_index, edge_index_readout, edge_attr, batch):
        indices = (x==0).nonzero().squeeze()
        h_nodes = self.conv(self.embed(x), edge_index, self.embed_ee(edge_attr), batch)
        x = h_nodes[indices]
        return x



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        #self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.word_embeddings = GraphTransformer(config)
        self.type_embeddings = nn.Embedding(11, config.hidden_size//5, padding_idx=0)

        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(config.age_vocab_size, config.hidden_size//5))

        self.time_embeddings = nn.Embedding(367, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(367, config.hidden_size//5))

        self.delta_embeddings = nn.Embedding(config.delta_size, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(config.delta_size, config.hidden_size//5))

        self.los_embeddings = nn.Embedding(1192, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(1192, config.hidden_size//5))

        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size//5). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size//5))



        self.seq_layers = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.acti = nn.GELU()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, los):
        word_embed = self.word_embeddings(nodes, edge_index, edge_index_readout, edge_attr, batch)
        type_embeddings = self.type_embeddings(type_ids)
        age_embed = self.age_embeddings(age_ids)
        los_embed = self.los_embeddings(los)

        time_embeddings = self.time_embeddings(time_ids)
        delta_embeddings = self.delta_embeddings(delta_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)


        word_embed = torch.reshape(word_embed, type_embeddings.shape)
        embeddings = torch.cat((word_embed, type_embeddings, posi_embeddings, age_embed, time_embeddings), dim=2)
        b, n, _ = embeddings.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = self.seq_layers(embeddings)
        embeddings = self.LayerNorm(embeddings)

        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

#%%

class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, attention_mask=None, los=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(age_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, los)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


#%%

class BertForMTR(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMTR, self).__init__(config)
        self.num_labels = 1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.gru = nn.GRU(config.hidden_size, config.hidden_size // 2, 1, batch_first = True, bidirectional=True)
        #self.gru = nn.Linear(config.hidden_size * 50, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.relu = nn.ReLU()
        self.apply(self.init_bert_weights)
    def forward(self, nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, attention_mask=None, labels=None, masks=None, los=None):
        _, pooled_output = self.bert(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids, delta_ids, type_ids, posi_ids, attention_mask,los,
                                     output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        #pooled_output = pooled_output * attention_mask.unsqueeze(-1)
        #pooled_output = torch.sum(pooled_output, axis=1) / torch.sum(attention_mask, axis=1).unsqueeze(-1)
        #pooled_output = torch.mean(_, axis=1)
        #pooled_output, x = self.gru(pooled_output)
        #pooled_output = self.gru(torch.flatten(pooled_output, start_dim=1))
        #pooled_output = self.relu(self.dropout(pooled_output))
        logits = self.classifier(pooled_output).squeeze(dim=1)
        bce_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')
        discr_supervised_loss = bce_logits_loss(logits, labels)

        return discr_supervised_loss, logits

#%%

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.age_vocab_size = config.get('age_vocab_size')
        self.delta_size = config.get('delta_size')
        self.graph_dropout_prob = config.get('graph_dropout_prob')

class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')

#%%

class GDSet(Dataset):
    def __init__(self, g):
        self.g = g

    def __getitem__(self, index):

        g = self.g[index]
        for i in range(len(g)):
          g[i]['posi_ids'] = i
        return g

    def __len__(self):
        return len(self.g)

# %% [cell 4]
import json
import os
import pickle
from pathlib import Path

from gtbehrt_lazy_dataset import GTBEHRTIndexSubset, load_gtbehrt_dataset, load_gtbehrt_pid_to_index

DATA_PATH = Path(os.environ.get("GTBEHRT_DATA_PATH", str(Path(__file__).resolve().parent / "sample_None" / "data")))
SEQ_LEN = int(os.environ.get("GTBEHRT_MAX_SEQ_LEN", "50"))
MAX_CODES_PER_VISIT = int(os.environ.get("GTBEHRT_MAX_CODES_PER_VISIT", "0")) or None
FEW_SHOTS = float(os.environ.get("GTBEHRT_FEW_SHOTS", "1.0"))
GTBEHRT_BATCH_SIZE = int(os.environ.get("GTBEHRT_BATCH_SIZE", "64"))
GTBEHRT_EPOCHS = int(os.environ.get("GTBEHRT_EPOCHS", "30"))
GTBEHRT_SEED = int(os.environ.get("GTBEHRT_RANDOM_SEED", "1"))
GTBEHRT_ACTION = os.environ.get("GTBEHRT_ACTION", "train")
GTBEHRT_MODEL_SAVE_PATH = os.environ.get("GTBEHRT_MODEL_SAVE_PATH", "models/v_behrt")
GTBEHRT_LOG_PATH = os.environ.get("GTBEHRT_LOG_PATH", "v_behrt_log_train.txt")
GTBEHRT_PREDS_DIR = Path(os.environ.get("GTBEHRT_PREDS_DIR", "preds"))
GTBEHRT_SPLITS_DIR = Path(os.environ["GTBEHRT_SPLITS_DIR"]) if os.environ.get("GTBEHRT_SPLITS_DIR") else None
GTBEHRT_COMBINED_PATH = Path(os.environ.get("GTBEHRT_COMBINED_PATH", str(DATA_PATH.parent / "pipeline_output.combined.train")))
GTBEHRT_CODE_VOCAB_PATH = Path(os.environ.get("GTBEHRT_CODE_VOCAB_PATH", str(DATA_PATH.parent / "code_vocab.pkl")))
GTBEHRT_TEST_METRICS_PATH = Path(os.environ.get("GTBEHRT_TEST_METRICS_PATH", "test_metrics.json"))
GTBEHRT_PROFILE_STEPS = int(os.environ.get("GTBEHRT_PROFILE_STEPS", "0"))
GTBEHRT_TIMING = os.environ.get("GTBEHRT_TIMING", "0") == "1" or GTBEHRT_PROFILE_STEPS > 0
GTBEHRT_TIMING_INTERVAL = int(os.environ.get("GTBEHRT_TIMING_INTERVAL", "100"))
print('loading:', DATA_PATH)
print('seq_len:', SEQ_LEN)
if DATA_PATH.exists():
    print('size_gb:', DATA_PATH.stat().st_size / 1024**3)

random.seed(GTBEHRT_SEED)
np.random.seed(GTBEHRT_SEED)
torch.manual_seed(GTBEHRT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GTBEHRT_SEED)

dataset, dataset_mode = load_gtbehrt_dataset(DATA_PATH, max_seq_len=SEQ_LEN, max_codes_per_visit=MAX_CODES_PER_VISIT)
print('dataset_mode:', dataset_mode)
print('max_codes_per_visit:', MAX_CODES_PER_VISIT)
print('patients:', len(dataset))
print('visits in first patient:', len(dataset[0]) if len(dataset) else 0)


def load_vocab_size(code_vocab_path: Path):
    if not code_vocab_path.exists():
        raise FileNotFoundError(f"Missing GT-BEHRT code vocab at {code_vocab_path}")
    with code_vocab_path.open('rb') as handle:
        code_vocab = pickle.load(handle)
    if not code_vocab:
        return 1
    return max(int(v) for v in code_vocab.values()) + 2


VOCAB_SIZE = load_vocab_size(GTBEHRT_CODE_VOCAB_PATH)
print('vocab_size:', VOCAB_SIZE)


def load_fixed_split_indices(splits_dir: Path, combined_path: Path):
    pid_to_index = load_gtbehrt_pid_to_index(combined_path)

    def indices_for(file_name: str):
        pids = torch.load(splits_dir / file_name)
        missing = [int(pid) for pid in pids if int(pid) not in pid_to_index]
        if missing:
            print(f"warning: {len(missing)} pids from {file_name} not found in GT-BEHRT dataset")
        return [pid_to_index[int(pid)] for pid in pids if int(pid) in pid_to_index]

    return indices_for('train_pids.pt'), indices_for('val_pids.pt'), indices_for('test_pids.pt')


if GTBEHRT_SPLITS_DIR is not None:
    train_index, val_index, test_index = load_fixed_split_indices(GTBEHRT_SPLITS_DIR, GTBEHRT_COMBINED_PATH)
    if FEW_SHOTS < 1:
        train_index = random.sample(list(train_index), max(1, int(len(train_index) * FEW_SHOTS)))
else:
    from sklearn.model_selection import ShuffleSplit
    rr = GTBEHRT_SEED
    rs = ShuffleSplit(n_splits=1, test_size=.20, random_state=rr)
    train_index, val_index, test_index = None, None, None
    for _, (train_index_tmp, test_index_tmp) in enumerate(rs.split(dataset)):
        rs2 = ShuffleSplit(n_splits=1, test_size=.125, random_state=rr)
        for _, (train_index_tmp2, val_index_tmp) in enumerate(rs2.split(train_index_tmp)):
            train_index = train_index_tmp[train_index_tmp2]
            if FEW_SHOTS < 1:
                train_index = random.sample(list(train_index), max(1, int(len(train_index) * FEW_SHOTS)))
            val_index = train_index_tmp[val_index_tmp]
            test_index = test_index_tmp

train_l = len(train_index)
val_l = len(val_index)
test_l = len(test_index)
number_output = 1

# %% [cell 6]
file_config = {
    'model_path': 'model/', # where to save model
    'model_name': 'CVDTransformer', # model name
    'file_name': 'log.txt',  # log path
}
#create_folder(file_config['model_path'])

global_params = {
    'max_seq_len': SEQ_LEN,
    'month': 1,
    'gradient_accumulation_steps': 1
}

optim_param = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

train_params = {
    'batch_size': GTBEHRT_BATCH_SIZE,
    'use_cuda': True,
    'max_len_seq': global_params['max_seq_len'],
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'data_len' : len(dataset),
    'train_data_len' : train_l,
    'val_data_len' : val_l,
    'test_data_len' : test_l,
    'epochs' : GTBEHRT_EPOCHS,
    'action' : GTBEHRT_ACTION
}

model_config = {
    'vocab_size': VOCAB_SIZE, # number of disease + symbols for word embedding
    'hidden_size': 108*5, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': 103, # number of vocab for age embedding
    'delta_size': 144, # number of vocab for age embedding
    'gender_vocab_size': 2,
    'ethnicity_vocab_size': 2,
    'race_vocab_size': 6,
    'num_labels':1,
    'feature_dict': VOCAB_SIZE,
    'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.2, # dropout rate
    'graph_dropout_prob': 0.2, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 12, # number of attention heads
    'attention_probs_dropout_prob': 0.2, # multi-head attention dropout rate
    'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
    'number_output' : number_output,
    'n_layers' : 3 - 1,
    'alpha' : 0.1
}

# %% [cell 7]
trainDSet = GTBEHRTIndexSubset(dataset, train_index)
valDSet = GTBEHRTIndexSubset(dataset, val_index)
testDSet = GTBEHRTIndexSubset(dataset, test_index)

# %% [cell 8]
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
conf = BertConfig(model_config)
behrt = BertForMTR(conf)

behrt = behrt.to(train_params['device'])

#models parameters
transformer_vars = [i for i in behrt.parameters()]

#optimizer
import transformers
optim_behrt = torch.optim.AdamW(transformer_vars, lr=3e-5)
#sched = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optim_behrt, 1000, 500*train_params['epochs'], 4, -1)

# %% [cell 9]
def _empty_timing_breakdown():
    return {
        "dataloader_wait": 0.0,
        "zero_grad": 0.0,
        "batch_graph": 0.0,
        "to_device": 0.0,
        "tensor_prep": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optim_step": 0.0,
    }


def _timing_summary(name, timings, steps):
    total = sum(float(v) for v in timings.values())
    lines = [f"{name} timing over {steps} steps (tracked {total:.2f}s):"]
    for key, value in timings.items():
        pct = (100.0 * value / total) if total > 0 else 0.0
        per_step = (value / steps) if steps > 0 else 0.0
        lines.append(f"  {key}: {value:.2f}s total | {per_step:.4f}s/step | {pct:.1f}%")
    return "\n".join(lines)


def _log_timing(name, timings, steps):
    summary = _timing_summary(name, timings, steps)
    print(summary)
    with open(GTBEHRT_LOG_PATH, 'a') as f:
        f.write(summary + "\n")


def run_epoch(e, trainload, device):
    tr_loss = 0
    start = time.time()
    timings = _empty_timing_breakdown()
    steps_run = 0
    behrt.train()
    train_iter = iter(trainload)
    while True:
        wait_start = time.perf_counter()
        try:
            data = next(train_iter)
        except StopIteration:
            break
        timings["dataloader_wait"] += time.perf_counter() - wait_start
        steps_run += 1

        section_start = time.perf_counter()
        optim_behrt.zero_grad()
        timings["zero_grad"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        timings["batch_graph"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        graph_batch = graph_batch.to(device)
        timings["to_device"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        nodes = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        seq_len = train_params['max_len_seq']
        age_ids = torch.reshape(graph_batch.age, [graph_batch.age.shape[0] // seq_len, seq_len])
        time_ids = torch.reshape(graph_batch.time, [graph_batch.time.shape[0] // seq_len, seq_len])
        delta_ids = torch.reshape(graph_batch.delta, [graph_batch.delta.shape[0] // seq_len, seq_len])
        type_ids = torch.reshape(graph_batch.adm_type, [graph_batch.adm_type.shape[0] // seq_len, seq_len])
        posi_ids = torch.reshape(graph_batch.posi_ids, [graph_batch.posi_ids.shape[0] // seq_len, seq_len])
        attMask = torch.reshape(graph_batch.mask_v, [graph_batch.mask_v.shape[0] // seq_len, seq_len])
        attMask = torch.cat((torch.ones((attMask.shape[0], 1)).to(device), attMask), dim=1)
        los = torch.reshape(graph_batch.los, [graph_batch.los.shape[0] // seq_len, seq_len])
        labels = torch.reshape(graph_batch.label, [graph_batch.label.shape[0] // seq_len, seq_len])[:, 0].float()
        masks = torch.reshape(graph_batch.mask, [graph_batch.mask.shape[0] // seq_len, seq_len])[:, 0]
        timings["tensor_prep"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        loss, logits = behrt(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids,delta_ids,type_ids,posi_ids,attMask, labels, masks, los)
        timings["forward"] += time.perf_counter() - section_start

        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']

        section_start = time.perf_counter()
        loss.backward()
        timings["backward"] += time.perf_counter() - section_start
        tr_loss += loss.item()
        if (steps_run - 1) % 500 == 0:
            print(loss.item())

        section_start = time.perf_counter()
        optim_behrt.step()
        timings["optim_step"] += time.perf_counter() - section_start
        del loss

        if GTBEHRT_TIMING and (steps_run % GTBEHRT_TIMING_INTERVAL == 0):
            print(_timing_summary(f"TRAIN epoch {e} partial", timings, steps_run))

        if GTBEHRT_PROFILE_STEPS and steps_run >= GTBEHRT_PROFILE_STEPS:
            print(f"Stopping train epoch {e} early after {steps_run} profiling steps")
            break
    cost = time.time() - start
    return tr_loss, cost, timings, steps_run
#%%

def train(trainload, valload, device):
    with open(GTBEHRT_LOG_PATH, 'w') as f:
            f.write('')
    best_val = math.inf
    for e in range(train_params["epochs"]):
        print("Epoch n" + str(e))
        train_loss, train_time_cost, train_timings, train_steps = run_epoch(e, trainload, device)
        val_loss, val_time_cost,pred, label, mask, val_timings, val_steps = eval(valload, False, device)
        train_denom = max(train_steps, 1)
        val_denom = max(val_steps, 1)
        train_loss = (train_loss * train_params['batch_size']) / train_denom
        val_loss = (val_loss * train_params['batch_size']) / val_denom
        print('TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
        with open(GTBEHRT_LOG_PATH, 'a') as f:
            f.write("Epoch n" + str(e) + '\n TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
            f.write('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost) + '\n')
        if GTBEHRT_TIMING:
            _log_timing(f"TRAIN epoch {e}", train_timings, train_steps)
            _log_timing(f"EVAL epoch {e}", val_timings, val_steps)
            with open(GTBEHRT_LOG_PATH, 'a') as f:
                f.write('\n')
        else:
            with open(GTBEHRT_LOG_PATH, 'a') as f:
                f.write('\n\n')
        print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost))
        if val_loss < best_val:
            print("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = behrt.module if hasattr(behrt, 'module') else behrt
            save_model(model_to_save.state_dict(), GTBEHRT_MODEL_SAVE_PATH)
            best_val = val_loss
    return train_loss, val_loss


#%%

def eval(_valload, saving, device):
    tr_loss = 0
    tr_g_loss = 0
    tr_d_un = 0
    tr_d_sup = 0
    temp_loss = 0
    start = time.time()
    timings = _empty_timing_breakdown()
    steps_run = 0
    all_logits = []
    all_labels = []
    all_masks = []
    behrt.eval()
    if saving:
        GTBEHRT_PREDS_DIR.mkdir(parents=True, exist_ok=True)
        with open(GTBEHRT_PREDS_DIR / "v_behrt_preds.csv", 'w') as f:
            f.write('')
        with open(GTBEHRT_PREDS_DIR / "v_behrt_labels.csv", 'w') as f:
            f.write('')
        with open(GTBEHRT_PREDS_DIR / "v_behrt_masks.csv", 'w') as f:
            f.write('')
    val_iter = iter(_valload)
    while True:
        wait_start = time.perf_counter()
        try:
            data = next(val_iter)
        except StopIteration:
            break
        timings["dataloader_wait"] += time.perf_counter() - wait_start
        steps_run += 1

        section_start = time.perf_counter()
        optim_behrt.zero_grad()
        timings["zero_grad"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        timings["batch_graph"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        graph_batch = graph_batch.to(device)
        timings["to_device"] += time.perf_counter() - section_start
        nodes = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        seq_len = train_params['max_len_seq']

        section_start = time.perf_counter()
        age_ids = torch.reshape(graph_batch.age, [graph_batch.age.shape[0] // seq_len, seq_len])
        time_ids = torch.reshape(graph_batch.time, [graph_batch.time.shape[0] // seq_len, seq_len])
        delta_ids = torch.reshape(graph_batch.delta, [graph_batch.delta.shape[0] // seq_len, seq_len])
        type_ids = torch.reshape(graph_batch.adm_type, [graph_batch.adm_type.shape[0] // seq_len, seq_len])
        posi_ids = torch.reshape(graph_batch.posi_ids, [graph_batch.posi_ids.shape[0] // seq_len, seq_len])
        attMask = torch.reshape(graph_batch.mask_v, [graph_batch.mask_v.shape[0] // seq_len, seq_len])
        attMask = torch.cat((torch.ones((attMask.shape[0], 1)).to(device), attMask), dim=1)
        los = torch.reshape(graph_batch.los, [graph_batch.los.shape[0] // seq_len, seq_len])
        labels = torch.reshape(graph_batch.label, [graph_batch.label.shape[0] // seq_len, seq_len])[:, 0].float()
        masks = torch.reshape(graph_batch.mask, [graph_batch.mask.shape[0] // seq_len, seq_len])[:, 0]
        timings["tensor_prep"] += time.perf_counter() - section_start

        section_start = time.perf_counter()
        loss, logits = behrt(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids,delta_ids,type_ids,posi_ids,attMask, labels, masks, los)
        timings["forward"] += time.perf_counter() - section_start

        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu()
        masks_cpu = masks.detach().cpu()
        all_logits.append(logits_cpu)
        all_labels.append(labels_cpu)
        all_masks.append(masks_cpu)

        if saving:
            with open(GTBEHRT_PREDS_DIR / "v_behrt_preds.csv", 'a') as f:
                pd.DataFrame(logits_cpu.numpy()).to_csv(f, header=False)
            with open(GTBEHRT_PREDS_DIR / "v_behrt_labels.csv", 'a') as f:
                pd.DataFrame(labels_cpu.numpy()).to_csv(f, header=False)
            with open(GTBEHRT_PREDS_DIR / "v_behrt_masks.csv", 'a') as f:
                pd.DataFrame(masks_cpu.numpy()).to_csv(f, header=False)

        tr_loss += loss.item()
        del loss

        if GTBEHRT_TIMING and (steps_run % GTBEHRT_TIMING_INTERVAL == 0):
            print(_timing_summary("EVAL partial", timings, steps_run))

        if GTBEHRT_PROFILE_STEPS and steps_run >= GTBEHRT_PROFILE_STEPS:
            print(f"Stopping eval early after {steps_run} profiling steps")
            break

    denom = max(steps_run, 1)
    print("TOTAL LOSS", (tr_loss * train_params['batch_size']) / denom)

    cost = time.time() - start
    if all_logits:
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        masks = torch.cat(all_masks, dim=0)
    else:
        logits = torch.empty(0)
        labels = torch.empty(0)
        masks = torch.empty(0)
    return tr_loss, cost, logits, labels, masks, timings, steps_run

#%%

def compute_test_metrics(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    truth = labels.detach().cpu().numpy().reshape(-1).astype(int)
    metrics = {
        "num_examples": int(len(truth)),
        "positive_examples": int(truth.sum()) if len(truth) else 0,
        "accuracy": float(skm.accuracy_score(truth, (probs >= 0.5).astype(int))) if len(truth) else None,
    }
    if len(set(truth.tolist())) > 1:
        metrics["auroc"] = float(skm.roc_auc_score(truth, probs))
        metrics["auprc"] = float(skm.average_precision_score(truth, probs))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return metrics


def write_test_metrics(metrics):
    GTBEHRT_TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GTBEHRT_TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")


def save_model(_model_dict, file_name):
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_model_dict, path)

# %% [cell 10]
def count_parameters(model):
  return sum(p.numel() for p in model.parameters())
count_parameters(behrt)

# %% [cell 11]
PRETRAINED_PATH = os.environ.get("GTBEHRT_PRETRAINED_PATH")

if PRETRAINED_PATH:
    pretrained_dict = torch.load(PRETRAINED_PATH, map_location=train_params['device'])
    model_dict = behrt.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    behrt.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)} pretrained tensors")
else:
    print("No pretrained checkpoint provided; training from scratch.")

# %% [cell 12]
print(train_params['max_len_seq'])
if train_params['action'] == 'train' or train_params['action'] == 'resume':
    trainload = GraphLoader(GDSet(trainDSet), batch_size=train_params['batch_size'], shuffle=False)
    valload = GraphLoader(GDSet(valDSet), batch_size=train_params['batch_size'], shuffle=False)

    train_loss, val_loss = train(trainload, valload, train_params['device'])
elif train_params['action'] == 'test':
    testload = GraphLoader(GDSet(testDSet), batch_size=train_params['batch_size'], shuffle=False)
    test_loss, test_time_cost, pred, label, mask, test_timings, test_steps = eval(testload, True, train_params['device'])
    metrics = compute_test_metrics(pred, label)
    metrics['loss'] = (test_loss * train_params['batch_size']) / max(len(testload), 1)
    metrics['time_seconds'] = float(test_time_cost)
    write_test_metrics(metrics)
    print(metrics)
else:
    raise ValueError(f"Unsupported GT-BEHRT action: {train_params['action']}")
