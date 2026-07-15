"""
Heterogeneous GNN (plan Stage 7) + temperature scaling (Fix 32).

Two node types keep their native feature dimensions (6 cortical, 4 subcortical)
and are projected to a shared hidden width before typed message passing, so the
model never sees zero padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, GraphNorm, global_mean_pool

from hetero_data import CT, SC, R_CC, R_SS, R_CS, R_SC

CORTICAL_IN = 6      # thickness, area, lGI, x, y, z
SUBCORTICAL_IN = 4   # volume, x, y, z


class HeteroGNN(nn.Module):
    """
    Normalization choice — do NOT change this back to GraphNorm without reading
    ------------------------------------------------------------------------
    The plan inherits GraphNorm from GAT v3, where it was adopted to cure
    BatchNorm instability. On THIS task it is actively destructive, and measurably
    so. GraphNorm centres each channel across the nodes WITHIN a graph:

        x_i <- (x_i - mean_j x_j) / std_j(x_j) * gamma + beta

    Every graph here has the same 96 fixed nodes and the diagnosis signal lives
    largely in subject-level magnitude (an ASD brain being globally thinner).
    Centring across nodes removes precisely that, and since the readout is
    global_mean_pool over the same nodes, mean_i(x_i) collapses toward the
    constant beta -- identical for every subject.

    Measured on a real fold, the across-subject SD of the pooled graph vector:
        GraphNorm 0.024   vs   BatchNorm 0.199   (8.4x)
    and swapping GraphNorm -> BatchNorm moved LOSO AUC 0.473 -> 0.551.

    BatchNorm normalizes across all nodes in the batch, preserving between-subject
    differences. Graph size is constant here, so GAT v3's instability argument for
    GraphNorm does not apply.
    """

    def __init__(self, hidden=64, heads=4, layers=2, dropout=0.3, norm="batch"):
        super().__init__()
        self.norm_kind = norm
        self.enc = nn.ModuleDict({
            CT: nn.Linear(CORTICAL_IN, hidden),
            SC: nn.Linear(SUBCORTICAL_IN, hidden),
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv({
                rel: GATv2Conv((hidden, hidden), hidden // heads, heads=heads,
                               edge_dim=1, add_self_loops=False, dropout=dropout)
                for rel in (R_CC, R_SS, R_CS, R_SC)
            }, aggr="sum")
            self.convs.append(conv)
            if norm == "graph":
                mk = lambda: GraphNorm(hidden)          # kept for the ablation only
            elif norm == "batch":
                mk = lambda: nn.BatchNorm1d(hidden)
            else:
                mk = lambda: nn.Identity()
            self.norms.append(nn.ModuleDict({CT: mk(), SC: mk()}))

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        # Calibration scalar. Fitted post-hoc on the inner validation split and
        # frozen; it must never be trained with the classification loss.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, data, logits_only=True):
        x = {CT: self.enc[CT](data[CT].x), SC: self.enc[SC](data[SC].x)}
        ei = {r: data[r].edge_index for r in (R_CC, R_SS, R_CS, R_SC)}
        ea = {r: data[r].edge_attr for r in (R_CC, R_SS, R_CS, R_SC)}

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, ei, edge_attr_dict=ea)
            x = {k: F.elu(norm[k](v, data[k].batch) if self.norm_kind == "graph"
                          else norm[k](v))
                 for k, v in x.items()}
            x = {k: F.dropout(v, p=self.dropout, training=self.training)
                 for k, v in x.items()}

        g = torch.cat([
            global_mean_pool(x[CT], data[CT].batch),
            global_mean_pool(x[SC], data[SC].batch),
        ], dim=1)
        logits = self.head(g)
        return logits if logits_only else logits / self.temperature

    @torch.no_grad()
    def predict_proba(self, data):
        """Calibrated P(ASD). Applies the fitted temperature."""
        return F.softmax(self.forward(data, logits_only=False), dim=1)[:, 1]


def fit_temperature(model, loader, device, max_iter=200):
    """Temperature scaling on a held-out validation split (Guo et al. 2017).

    Optimises ONLY the temperature against NLL; all other weights are frozen and
    the decision ranking (hence AUC) is unchanged by construction.
    """
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits.append(model(b))
            ys.append(b.y)
    logits = torch.cat(logits)
    ys = torch.cat(ys)

    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), ys)
        loss.backward()
        return loss

    opt.step(closure)
    model.temperature.fill_(float(log_t.exp().item()))
    return float(model.temperature.item())


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE with equal-width confidence bins."""
    import numpy as np
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf = np.maximum(probs, 1 - probs)
    pred = (probs >= 0.5).astype(int)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)
