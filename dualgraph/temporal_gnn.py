"""
Temporal node encoder + functional GNN (the one idea that can break the FC ceiling).

Rationale: every node/edge feature so far is a function of the STATIC fc_z, so by
the section-4.3 ceiling a GNN over them cannot exceed the edge-SVM (0.658). A
temporal encoder reads the RAW [T,111] time-series -- strictly more information
than its correlation summary -- so it is the one component that can, in principle,
add signal. Cost: the raw BOLD is where motion lives, so G1/G3 are mandatory
(train_temporal.py runs them on the out-of-fold predictions).

Design choices for THIS data (short T=116, N=780, motion-confounded):
  * dilated 1D-CNN, NOT an LSTM -- short sequences, less overfitting.
  * weights SHARED across all 111 nodes (one small encoder, applied per node).
  * global average pool over time -- averages out transient motion spikes.
  * edges stay the proven fc_z topology (top-k) with edge_attr = fc_z, unchanged.

Per-subject graph:
  data.x          [111, T]   node time-series (encoded inside forward)
  data.edge_index [2, E]     top-k on |fc_z|, symmetrised
  data.edge_attr  [E, 1]     fc_z value on each edge
  data.y          [1]        ASD=1 / TD=0
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


def build_edges(fc_z: np.ndarray, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k per node on |fc_z|, symmetrised. Returns (edge_index[2,E], edge_attr[E,1])."""
    W = np.abs(fc_z).copy()
    np.fill_diagonal(W, 0.0)
    nn_idx = np.argsort(-W, axis=1)[:, :k]                 # [N, k]
    N = fc_z.shape[0]
    src = np.repeat(np.arange(N), k)
    dst = nn_idx.ravel()
    # symmetrise: keep union of (i->j) and (j->i)
    a = np.vstack([src, dst])
    b = np.vstack([dst, src])
    ei = np.unique(np.hstack([a, b]), axis=1)
    ew = fc_z[ei[0], ei[1]].astype(np.float32)[:, None]
    return torch.tensor(ei, dtype=torch.long), torch.tensor(ew, dtype=torch.float32)


def make_graph(ts: np.ndarray, fc_z: np.ndarray, label: int, k: int = 10) -> Data:
    ei, ew = build_edges(fc_z, k)
    x = torch.tensor(ts.T, dtype=torch.float32)            # [N, T]
    return Data(x=x, edge_index=ei, edge_attr=ew,
                y=torch.tensor([label], dtype=torch.long))


class TemporalNodeEncoder(nn.Module):
    """Shared dilated 1D-CNN: [num_nodes, T] -> [num_nodes, out_dim]."""

    def __init__(self, out_dim: int = 16, channels: tuple[int, ...] = (8, 16),
                 kernel: int = 5, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_c, dil = 1, 1
        for c in channels:
            layers += [nn.Conv1d(in_c, c, kernel, padding="same", dilation=dil),
                       nn.BatchNorm1d(c), nn.ELU(), nn.Dropout(dropout)]
            in_c, dil = c, dil * 2
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(in_c, out_dim)

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x_ts.unsqueeze(1))                    # [num_nodes, C, T]
        h = h.mean(dim=-1)                                 # global avg pool over time
        return self.proj(h)                                # [num_nodes, out_dim]


class TemporalFuncGNN(nn.Module):
    def __init__(self, enc_dim: int = 16, hidden: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.enc = TemporalNodeEncoder(out_dim=enc_dim, dropout=dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            in_d = enc_dim if i == 0 else hidden
            self.convs.append(GATv2Conv(in_d, hidden // heads, heads=heads,
                                        edge_dim=1, add_self_loops=False,
                                        dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, data: Data) -> torch.Tensor:
        x = self.enc(data.x)                               # [num_nodes, enc_dim]
        ei, ea = data.edge_index, data.edge_attr
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, ei, edge_attr=ea)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, data.batch)
        return self.head(g)
