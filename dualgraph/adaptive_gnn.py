"""
Adaptive functional GNN with robust4 node features.

Node features : robust4 [111,4] = [strength, eigen-centrality, within-module-z,
                participation], all derived from fc_z (see node_features.py).
Edge channels : two relations, selectable by variant --
  fc    : top-k on |fc_z|, edge_attr = fc_z            (the proven per-subject topology)
  adapt : SubjectAdaptiveGraph -- project node features, L2-normalise, cosine
          similarity, top-k per node; edge_attr = the similarity     (plan 4.1)

Variants:
  v1 "fc"    fc_z edges only              -- rung 1b, the honest GNN baseline
  v2 "adapt" learned edges only           -- pure SubjectAdaptiveGraph
  v3 "dual"  both relations, messages summed  -- recommended primary

Why v3 is primary: robust4 is itself a lossy function of fc_z, so inferring
topology *only* from it (v2) throws away the connectivity we already hold. v3
keeps the proven fc_z structure and lets the learned relation refine it.

Normalisation is BatchNorm, never GraphNorm: measured on this project, GraphNorm
collapsed the pooled-vector SD (0.024 vs 0.199) and moved LOSO AUC 0.473 -> 0.551.

The adaptive topology is built inside forward() from the *current* node features,
so it varies per subject by construction; assert_adaptive_varies() is the
load-bearing unit test (the earlier defect was one identical edge tensor for every
subject, std = 0.000000).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

N_NODES = 111


def fc_edges(fc_z: np.ndarray, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k per node on |fc_z|, symmetrised. edge_attr = fc_z on each edge."""
    W = np.abs(fc_z).copy()
    np.fill_diagonal(W, 0.0)
    idx = np.argsort(-W, axis=1)[:, :k]
    n = fc_z.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx.ravel()
    ei = np.unique(np.hstack([np.vstack([src, dst]), np.vstack([dst, src])]), axis=1)
    ew = fc_z[ei[0], ei[1]].astype(np.float32)[:, None]
    return torch.tensor(ei, dtype=torch.long), torch.tensor(ew, dtype=torch.float32)


def make_graph(x_robust: np.ndarray, fc_z: np.ndarray, label: int,
               k: int = 10) -> Data:
    ei, ew = fc_edges(fc_z, k)
    return Data(x=torch.tensor(x_robust, dtype=torch.float32),
                edge_index=ei, edge_attr=ew,
                y=torch.tensor([label], dtype=torch.long))


def adaptive_edges(h: torch.Tensor, k: int, n_nodes: int = N_NODES
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-subject cosine-similarity top-k graph from projected node features.

    h: [B*n_nodes, d] (already L2-normalised). Node count is fixed, so the batch
    reshapes cleanly and no to_dense_batch padding is needed.
    Returns (edge_index [2,E], edge_attr [E,1]) with correct per-graph offsets.
    """
    B = h.shape[0] // n_nodes
    hb = h.view(B, n_nodes, -1)
    S = torch.bmm(hb, hb.transpose(1, 2))                      # [B,N,N] cosine
    eye = torch.eye(n_nodes, dtype=torch.bool, device=h.device)
    S = S.masked_fill(eye.unsqueeze(0), float("-inf"))         # no self-loops
    val, idx = S.topk(k, dim=-1)                               # [B,N,k]
    off = (torch.arange(B, device=h.device) * n_nodes).view(B, 1, 1)
    src = torch.arange(n_nodes, device=h.device).view(1, n_nodes, 1).expand(B, n_nodes, k) + off
    dst = idx + off
    ei = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
    ea = val.reshape(-1, 1)
    return ei, ea


class AdaptiveFuncGNN(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.3, k_adapt: int = 10,
                 d_proj: int = 16, variant: str = "dual", readout: str = "meanmax"):
        super().__init__()
        assert variant in ("fc", "adapt", "dual")
        assert readout in ("mean", "meanmax")
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.variant, self.k_adapt, self.dropout = variant, k_adapt, dropout
        self.readout = readout

        # FIX 2: mean-pool alone dilutes localised signal by ~1/111; concat(mean,max)
        # keeps "some region is strongly abnormal" alongside the average.
        pool_mult = 2 if readout == "meanmax" else 1

        # FIX 1 support: in_dim may now be 111 (each node's full fc_z row) instead of
        # 4 (robust4). A wide raw input needs a bottleneck + dropout, otherwise the
        # parameter count is what killed the temporal encoder.
        self.enc = (nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(),
                                  nn.Dropout(dropout))
                    if in_dim > 16 else nn.Linear(in_dim, hidden))
        self.proj = nn.Linear(in_dim, d_proj)                  # adaptive-graph projector

        self.conv_fc = nn.ModuleList()
        self.conv_ad = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if variant in ("fc", "dual"):
                self.conv_fc.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                              edge_dim=1, add_self_loops=False,
                                              dropout=dropout))
            if variant in ("adapt", "dual"):
                self.conv_ad.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                              edge_dim=1, add_self_loops=False,
                                              dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(nn.Linear(pool_mult * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, data: Data, return_pooled: bool = False):
        x0 = data.x
        x = self.enc(x0)
        ei_fc, ea_fc = data.edge_index, data.edge_attr

        ei_ad = ea_ad = None
        if self.variant in ("adapt", "dual"):
            h = F.normalize(self.proj(x0), p=2, dim=-1)
            ei_ad, ea_ad = adaptive_edges(h, self.k_adapt)

        n_layers = len(self.norms)
        for i in range(n_layers):
            msgs = []
            if self.variant in ("fc", "dual"):
                msgs.append(self.conv_fc[i](x, ei_fc, edge_attr=ea_fc))
            if self.variant in ("adapt", "dual"):
                msgs.append(self.conv_ad[i](x, ei_ad, edge_attr=ea_ad))
            x = sum(msgs) if len(msgs) > 1 else msgs[0]
            x = F.elu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, data.batch)
        if self.readout == "meanmax":
            g = torch.cat([g, global_max_pool(x, data.batch)], dim=1)
        logits = self.head(g)
        return (logits, g) if return_pooled else logits

    @torch.no_grad()
    def adaptive_edge_index(self, data: Data) -> torch.Tensor:
        """Expose the learned topology for the unit test."""
        h = F.normalize(self.proj(data.x), p=2, dim=-1)
        return adaptive_edges(h, self.k_adapt)[0]


@torch.no_grad()
def assert_adaptive_varies(model: AdaptiveFuncGNN, loader, device,
                           n_subjects: int = 50) -> float:
    """LOAD-BEARING TEST (plan 4.1/5.1): the adaptive topology must differ across
    subjects. The earlier null was caused by one identical edge tensor shared by
    every subject (std = 0.000000). Returns the across-subject std of the sorted
    neighbour-index signature; raises if it is zero.
    """
    model.eval()
    sigs = []
    for b in loader:
        b = b.to(device)
        ei = model.adaptive_edge_index(b)
        B = b.num_graphs
        per = ei.shape[1] // B
        for g in range(B):
            dst = ei[1, g * per:(g + 1) * per] - g * N_NODES
            sigs.append(torch.sort(dst.float()).values.cpu().numpy())
            if len(sigs) >= n_subjects:
                break
        if len(sigs) >= n_subjects:
            break
    S = np.stack(sigs)
    std = float(S.std(axis=0).mean())
    if std == 0.0:
        raise AssertionError(
            "adaptive edge_index is IDENTICAL across subjects (std=0.000000) -- "
            "this is the exact defect that produced the earlier null; the adaptive "
            "relation is vacuous.")
    return std
