"""
Stages 7-8: dual-graph fusion and cross-attention, plus the rung 1a / MLP controls.

Node counts are FIXED (96 structural, 111 functional), so this module works on dense
[B, N, F] tensors and builds edge_index with per-graph offsets inside forward. That
avoids PyG batching entirely and makes the cross-attention reshape (x.view(B, N, d))
safe by construction -- the plan flags a scrambled batch vector as a real hazard.

ARMS
  FuncArm   : fcrow node features [111,111] -> Linear(111,64)+ELU+Dropout
              relations: fc_z top-k  AND  SubjectAdaptiveGraph  (variant "dual")
  StructArm : 68 cortical [thickness,area,lGI,x,y,z] + 28 subcortical [volume,x,y,z]
              type-specific Linear(6,64) / Linear(4,64), NO zero padding (plan Fix 11)
              relation: SubjectAdaptiveGraph only -- the structural covariation graph
              is group-level, which is the std=0.000000 defect we are avoiding

RUNGS
  1a   SingleArm(struct)                    1b   SingleArm(func)
  1a-MLP / 1b-MLP  same encoder + readout,每 GATv2Conv replaced by per-node
                   Linear(64,64): identical depth, NO message passing (plan section 7
                   parity rule -- do not flatten, that would win on capacity)
  2    FusionNet     concat(pool(S), pool(F)) -> head          (no attention)
  3    CrossAttn A   Q=S, K=V=F  -> [B,96,64]   sMRI queries fMRI  (the novelty)
  4    CrossAttn B   Q=F, K=V=S  -> [B,111,64]  CAS-GNN direction

Fixed settings carried from the experiments: layers=2, hidden=64, heads=4,
dropout=0.3, k=20, BatchNorm (never GraphNorm), readout concat(mean,max).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

N_FUNC, N_CORT, N_SUB = 111, 68, 28
N_STRUCT = N_CORT + N_SUB


# ------------------------------------------------------------------ edge builders
def batched_topk_edges(W: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,N,N] dense weights -> (edge_index[2,E], edge_attr[E,1]) with graph offsets.
    Selection on |W|; the returned attribute is the SIGNED weight."""
    B, N, _ = W.shape
    A = W.abs().clone()
    eye = torch.eye(N, dtype=torch.bool, device=W.device)
    A = A.masked_fill(eye.unsqueeze(0), -float("inf"))
    _, idx = A.topk(k, dim=-1)                                    # [B,N,k]
    off = (torch.arange(B, device=W.device) * N).view(B, 1, 1)
    src = torch.arange(N, device=W.device).view(1, N, 1).expand(B, N, k) + off
    dst = idx + off
    ei = torch.stack([src.reshape(-1), dst.reshape(-1)], 0)
    ea = torch.gather(W, 2, idx).reshape(-1, 1)
    return ei, ea


def adaptive_edges(h: torch.Tensor, k: int, n_nodes: int
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """SubjectAdaptiveGraph: cosine top-k over L2-normalised projected features.
    h: [B*n_nodes, d] (normalised). Varies per subject by construction."""
    B = h.shape[0] // n_nodes
    hb = h.view(B, n_nodes, -1)
    S = torch.bmm(hb, hb.transpose(1, 2))
    eye = torch.eye(n_nodes, dtype=torch.bool, device=h.device)
    S = S.masked_fill(eye.unsqueeze(0), -float("inf"))
    val, idx = S.topk(k, dim=-1)
    off = (torch.arange(B, device=h.device) * n_nodes).view(B, 1, 1)
    src = torch.arange(n_nodes, device=h.device).view(1, n_nodes, 1).expand(B, n_nodes, k) + off
    ei = torch.stack([src.reshape(-1), (idx + off).reshape(-1)], 0)
    return ei, val.reshape(-1, 1)


def readout(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """[B*N,d] -> [B,2d] concat(mean,max). Measured +0.029 over mean alone."""
    B = x.shape[0] // n_nodes
    xb = x.view(B, n_nodes, -1)
    return torch.cat([xb.mean(1), xb.max(1).values], dim=1)


# ------------------------------------------------------------------ trunk
class GraphTrunk(nn.Module):
    """L layers of (message passing | per-node Linear) + BatchNorm + ELU + Dropout.

    mlp=True is the parity control: identical depth and per-node transform, message
    passing removed, so any GNN-vs-MLP gap is neighbour aggregation and nothing else.
    """

    def __init__(self, hidden=64, heads=4, layers=2, dropout=0.3,
                 use_fc=True, use_adapt=True, mlp=False):
        super().__init__()
        self.mlp, self.dropout = mlp, dropout
        self.use_fc, self.use_adapt = use_fc, use_adapt
        self.fc_convs, self.ad_convs, self.lins = (nn.ModuleList() for _ in range(3))
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if mlp:
                self.lins.append(nn.Linear(hidden, hidden))
            else:
                if use_fc:
                    self.fc_convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                                   edge_dim=1, add_self_loops=False,
                                                   dropout=dropout))
                if use_adapt:
                    self.ad_convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                                   edge_dim=1, add_self_loops=False,
                                                   dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden))

    def forward(self, x, ei_fc=None, ea_fc=None, ei_ad=None, ea_ad=None):
        for i, norm in enumerate(self.norms):
            if self.mlp:
                x = self.lins[i](x)
            else:
                msgs = []
                if self.use_fc:
                    msgs.append(self.fc_convs[i](x, ei_fc, edge_attr=ea_fc))
                if self.use_adapt:
                    msgs.append(self.ad_convs[i](x, ei_ad, edge_attr=ea_ad))
                x = sum(msgs) if len(msgs) > 1 else msgs[0]
            x = F.elu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ------------------------------------------------------------------ arms
class FuncArm(nn.Module):
    """fcrow functional arm -> node states [B*111, hidden]."""

    def __init__(self, hidden=64, heads=4, layers=2, dropout=0.3, k=20,
                 d_proj=16, mlp=False):
        super().__init__()
        self.k, self.mlp = k, mlp
        self.enc = nn.Sequential(nn.Linear(N_FUNC, hidden), nn.ELU(), nn.Dropout(dropout))
        self.proj = nn.Linear(N_FUNC, d_proj)
        self.trunk = GraphTrunk(hidden, heads, layers, dropout,
                                use_fc=True, use_adapt=True, mlp=mlp)

    def forward(self, x_f, fc):
        B, N, _ = x_f.shape
        flat = x_f.reshape(B * N, -1)
        h = self.enc(flat)
        if self.mlp:
            return self.trunk(h)
        ei_fc, ea_fc = batched_topk_edges(fc, self.k)
        p = F.normalize(self.proj(flat), p=2, dim=-1)
        ei_ad, ea_ad = adaptive_edges(p, self.k, N)
        return self.trunk(h, ei_fc, ea_fc, ei_ad, ea_ad)


class StructArm(nn.Module):
    """Structural arm: type-specific encoders, adaptive edges only (the covariation
    graph is group-level = the defect). -> node states [B*96, hidden]."""

    def __init__(self, hidden=64, heads=4, layers=2, dropout=0.3, k=20,
                 d_proj=16, mlp=False, in_cort=6, in_sub=4):
        super().__init__()
        self.k, self.mlp = k, mlp
        self.enc_c = nn.Linear(in_cort, hidden)
        self.enc_s = nn.Linear(in_sub, hidden)
        self.proj_c = nn.Linear(in_cort, d_proj)
        self.proj_s = nn.Linear(in_sub, d_proj)
        self.trunk = GraphTrunk(hidden, heads, layers, dropout,
                                use_fc=False, use_adapt=True, mlp=mlp)

    def forward(self, x_c, x_s):
        B = x_c.shape[0]
        h = torch.cat([self.enc_c(x_c), self.enc_s(x_s)], dim=1)      # [B,96,hidden]
        h = h.reshape(B * N_STRUCT, -1)
        if self.mlp:
            return self.trunk(h)
        p = torch.cat([self.proj_c(x_c), self.proj_s(x_s)], dim=1).reshape(B * N_STRUCT, -1)
        p = F.normalize(p, p=2, dim=-1)
        ei_ad, ea_ad = adaptive_edges(p, self.k, N_STRUCT)
        return self.trunk(h, None, None, ei_ad, ea_ad)


# ------------------------------------------------------------------ rungs
class SingleArmNet(nn.Module):
    """Rungs 1a / 1b and their MLP parity controls."""

    def __init__(self, modality="func", hidden=64, dropout=0.3, mlp=False, **kw):
        super().__init__()
        self.modality = modality
        self.n = N_FUNC if modality == "func" else N_STRUCT
        Arm = FuncArm if modality == "func" else StructArm
        self.arm = Arm(hidden=hidden, dropout=dropout, mlp=mlp, **kw)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, b, return_pooled=False):
        h = (self.arm(b["x_f"], b["fc"]) if self.modality == "func"
             else self.arm(b["x_c"], b["x_s"]))
        g = readout(h, self.n)
        return (self.head(g), g) if return_pooled else self.head(g)


class FusionNet(nn.Module):
    """Rung 2 -- dual-graph concat fusion, NO attention."""

    def __init__(self, hidden=64, dropout=0.3, mlp=False, **kw):
        super().__init__()
        self.f = FuncArm(hidden=hidden, dropout=dropout, mlp=mlp, **kw)
        self.s = StructArm(hidden=hidden, dropout=dropout, mlp=mlp, **kw)
        self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, b, return_pooled=False):
        gf = readout(self.f(b["x_f"], b["fc"]), N_FUNC)
        gs = readout(self.s(b["x_c"], b["x_s"]), N_STRUCT)
        g = torch.cat([gs, gf], dim=1)
        return (self.head(g), g) if return_pooled else self.head(g)


class CrossAttnNet(nn.Module):
    """Rungs 3 / 4 -- cross-attention between the two graphs.

    direction "A": Q=S, K=V=F -> [B,96,d]   sMRI queries fMRI (the novelty)
    direction "B": Q=F, K=V=S -> [B,111,d]  CAS-GNN direction
    Q length != KV length is fine; output length = Q length.
    """

    def __init__(self, direction="A", hidden=64, heads=4, dropout=0.3, mlp=False, **kw):
        super().__init__()
        assert direction in ("A", "B")
        self.direction = direction
        self.f = FuncArm(hidden=hidden, dropout=dropout, mlp=mlp, **kw)
        self.s = StructArm(hidden=hidden, dropout=dropout, mlp=mlp, **kw)
        self.mha = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads,
                                         dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, b, return_pooled=False):
        B = b["x_f"].shape[0]
        Fn = self.f(b["x_f"], b["fc"]).view(B, N_FUNC, -1)
        Sn = self.s(b["x_c"], b["x_s"]).view(B, N_STRUCT, -1)
        q, kv, n = ((Sn, Fn, N_STRUCT) if self.direction == "A" else (Fn, Sn, N_FUNC))
        att, _ = self.mha(q, kv, kv, need_weights=False)               # [B,n,hidden]
        g = torch.cat([att.mean(1), att.max(1).values], dim=1)
        return (self.head(g), g) if return_pooled else self.head(g)


def build_model(rung: str, **kw) -> nn.Module:
    """rung in {1a, 1b, 1a-mlp, 1b-mlp, 2, 3, 4}"""
    r = rung.lower()
    if r in ("1a", "1a-mlp"):
        return SingleArmNet("struct", mlp=r.endswith("mlp"), **kw)
    if r in ("1b", "1b-mlp"):
        return SingleArmNet("func", mlp=r.endswith("mlp"), **kw)
    if r == "2":
        return FusionNet(**kw)
    if r in ("3", "4"):
        return CrossAttnNet(direction="A" if r == "3" else "B", **kw)
    raise ValueError(f"unknown rung {rung}")
