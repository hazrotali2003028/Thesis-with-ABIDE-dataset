"""
Single-fold pilot: does a subject-adaptive graph beat the fixed group-level one?

Holds out ONE site (default NYU, the largest test set at 173 subjects) and runs,
on identical folds and identical harmonized features:

    hetero_fixed      current design: group-level covariation adjacency, tau=0.3
    hetero_adaptive   per-subject top-k graph learned from node embeddings
    logreg/svm/rf/xgboost/mlp/dummy   tabular floor on the same 232 features

This is a PILOT, not a result. One fold, no nested hyperparameter search, so the
numbers here are noisier than train_nested_loso.py's and must not be quoted as
LOSO performance. It answers one question: is the adaptive graph worth a full run?

Run:
    python pilot_adaptive_graph.py --site NYU
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool

from hetero_data import build_fold, harmonize_and_scale, CT, SC, R_CC, R_SS, R_CS, R_SC
from hetero_gnn import HeteroGNN

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")
COORDS = os.path.join(ROOT, "features", "node_coords.csv")

SEEDS = [42, 123, 456]
CORTICAL_IN, SUBCORTICAL_IN = 6, 4

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-adaptive graph
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_edges(h_src, batch_src, h_dst, batch_dst, k, self_loops=False):
    """Top-k cosine-similarity edges, computed WITHIN each subject.

    Restricting edges to WITHIN a subject is the whole point. PyG hands us every
    subject's nodes concatenated, so a global h @ h.T lets top-k select nodes
    belonging to a DIFFERENT subject's brain -- measured at 50% of edges for a
    2-subject batch and ~97% at batch_size=32.

    Rather than build the full (B*nodes, B*nodes) similarity and mask the
    cross-subject blocks to -inf (that matrix is ~1.2 GB at batch 256 and OOMs a
    4 GB GPU), we compute top-k per subject on its own (nodes, nodes) block. Same
    result, O(B * nodes^2) memory instead of O((B*nodes)^2).

    Returns edge_index (2, E) and edge_weight (E, 1) in src->dst convention,
    with node indices GLOBAL to the batch.
    """
    device = h_src.device
    src_glob = torch.arange(h_src.shape[0], device=device)
    dst_glob = torch.arange(h_dst.shape[0], device=device)
    src_out, dst_out, w_out = [], [], []

    for b in torch.unique(batch_src):
        sm = batch_src == b
        dm = batch_dst == b
        hs, hd = h_src[sm], h_dst[dm]
        gs, gd = src_glob[sm], dst_glob[dm]

        sim = hs @ hd.T                                       # (ns, nd)
        # Drop self-edges only when src and dst are the SAME node set (within
        # type). For cross-type (68 vs 28 nodes) self_loops=True and shapes
        # differ, so there is no diagonal to remove.
        if not self_loops and hs.shape[0] == hd.shape[0]:
            n = sim.shape[0]
            diag = torch.arange(n, device=device)
            sim[diag, diag] = float("-inf")

        k_eff = min(k, sim.shape[1])
        vals, idx = torch.topk(sim, k=k_eff, dim=1)           # (ns, k_eff)
        rows = torch.arange(sim.shape[0], device=device).unsqueeze(1).expand_as(idx)
        src_out.append(gs[rows.reshape(-1)])
        dst_out.append(gd[idx.reshape(-1)])
        w_out.append(vals.reshape(-1))

    ei = torch.stack([torch.cat(src_out), torch.cat(dst_out)])
    ew = torch.cat(w_out).unsqueeze(1)
    keep = torch.isfinite(ew.squeeze(1))
    return ei[:, keep], ew[keep]


class AdaptiveHeteroGNN(nn.Module):
    """HeteroGNN with the fixed adjacency replaced by a learned per-subject graph.

    Everything downstream of edge construction -- encoder widths, GATv2 layers,
    BatchNorm, mean-pool readout, head -- is byte-identical to HeteroGNN, so any
    difference in AUC is attributable to the graph and nothing else.
    """

    def __init__(self, hidden=64, heads=4, layers=2, dropout=0.3,
                 k=5, proj_dim=32):
        super().__init__()
        self.k = k
        self.enc = nn.ModuleDict({
            CT: nn.Linear(CORTICAL_IN, hidden),
            SC: nn.Linear(SUBCORTICAL_IN, hidden),
        })
        # Graph learners project into a SHARED space so cortical and subcortical
        # nodes are comparable for the cross-type relation.
        self.proj = nn.ModuleDict({
            CT: nn.Linear(CORTICAL_IN, proj_dim),
            SC: nn.Linear(SUBCORTICAL_IN, proj_dim),
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv({
                rel: GATv2Conv((hidden, hidden), hidden // heads, heads=heads,
                               edge_dim=1, add_self_loops=False, dropout=dropout)
                for rel in (R_CC, R_SS, R_CS, R_SC)
            }, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                CT: nn.BatchNorm1d(hidden), SC: nn.BatchNorm1d(hidden)}))

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        # Post-hoc calibration scalar; fitted on the inner validation split and
        # frozen, exactly as in HeteroGNN. Never trained with the class loss.
        self.register_buffer("temperature", torch.ones(1))

    def build_graph(self, data):
        hc = F.normalize(self.proj[CT](data[CT].x), p=2, dim=1)
        hs = F.normalize(self.proj[SC](data[SC].x), p=2, dim=1)
        bc, bs = data[CT].batch, data[SC].batch

        ei_cc, ew_cc = adaptive_edges(hc, bc, hc, bc, self.k)
        ei_ss, ew_ss = adaptive_edges(hs, bs, hs, bs, self.k)
        ei_cs, ew_cs = adaptive_edges(hc, bc, hs, bs, self.k, self_loops=True)
        ei_sc, ew_sc = adaptive_edges(hs, bs, hc, bc, self.k, self_loops=True)
        return ({R_CC: ei_cc, R_SS: ei_ss, R_CS: ei_cs, R_SC: ei_sc},
                {R_CC: ew_cc, R_SS: ew_ss, R_CS: ew_cs, R_SC: ew_sc})

    def forward(self, data, logits_only=True):
        ei, ea = self.build_graph(data)
        x = {CT: self.enc[CT](data[CT].x), SC: self.enc[SC](data[SC].x)}
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, ei, edge_attr_dict=ea)
            x = {k: F.elu(norm[k](v)) for k, v in x.items()}
            x = {k: F.dropout(v, p=self.dropout, training=self.training)
                 for k, v in x.items()}
        g = torch.cat([global_mean_pool(x[CT], data[CT].batch),
                       global_mean_pool(x[SC], data[SC].batch)], dim=1)
        logits = self.head(g)
        return logits if logits_only else logits / self.temperature

    @torch.no_grad()
    def predict_proba(self, data):
        """Calibrated P(ASD). Applies the fitted temperature."""
        return F.softmax(self.forward(data, logits_only=False), dim=1)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Training (mirrors train_nested_loso.run_epoch / train_one)
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, device, opt=None, class_w=None):
    train = opt is not None
    model.train(train)
    probs, ys = [], []
    for b in loader:
        b = b.to(device)
        with torch.set_grad_enabled(train):
            logits = model(b)
            loss = F.cross_entropy(logits, b.y, weight=class_w)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        probs.append(F.softmax(logits, 1)[:, 1].detach().cpu())
        ys.append(b.y.detach().cpu())
    probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    return auc, probs, ys


def train_gnn(kind, tr_graphs, va_graphs, te_graphs, seed, device,
              epochs, patience, k):
    set_seed(seed)
    tr_loader = DataLoader(tr_graphs, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_graphs, batch_size=128)
    te_loader = DataLoader(te_graphs, batch_size=128)

    ys = np.array([int(g.y) for g in tr_graphs])
    w = torch.tensor([len(ys) / (2 * max((ys == 0).sum(), 1)),
                      len(ys) / (2 * max((ys == 1).sum(), 1))],
                     dtype=torch.float32).to(device)

    if kind == "fixed":
        model = HeteroGNN(hidden=64, heads=4, layers=2, dropout=0.3).to(device)
    else:
        model = AdaptiveHeteroGNN(hidden=64, heads=4, layers=2, dropout=0.3,
                                  k=k).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5,
                                                       patience=5)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        run_epoch(model, tr_loader, device, opt, w)
        va_auc, _, _ = run_epoch(model, va_loader, device)
        sched.step(va_auc if not np.isnan(va_auc) else 0.0)
        if va_auc > best:
            best, bad = va_auc, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    te_auc, _, _ = run_epoch(model, te_loader, device)
    return te_auc, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="NYU")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    coords = pd.read_csv(COORDS)

    site = df["SITE_ID"].to_numpy()
    te_mask = site == args.site
    tr_mask = ~te_mask
    print(f"device={device}  holdout={args.site}  "
          f"train={tr_mask.sum()}  test={te_mask.sum()}")

    t0 = time.time()
    tr_graphs, te_graphs, info = build_fold(df, tr_mask, te_mask, coords, tau=0.3)
    print(f"fixed graph: cc={info['edges_cc']} ss={info['edges_ss']} "
          f"cs={info['edges_cs']} (density cc={info['density_cc']:.3f})")

    # Validation split carved from TRAINING sites only.
    tr_df = df[tr_mask].reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    i_tr, i_va = next(sss.split(tr_df, tr_df["label"]))
    sub_tr = [tr_graphs[i] for i in i_tr]
    sub_va = [tr_graphs[i] for i in i_va]
    print(f"inner split: train={len(sub_tr)} val={len(sub_va)}")

    rows = []

    # ── GNNs ──
    for kind in ("fixed", "adaptive"):
        for seed in SEEDS:
            auc, va = train_gnn(kind, sub_tr, sub_va, te_graphs, seed, device,
                                args.epochs, args.patience, args.k)
            rows.append({"model": f"hetero_{kind}", "seed": seed,
                         "test_auc": auc, "val_auc": va})
            print(f"  hetero_{kind:8s} seed={seed:4d}  val={va:.4f}  "
                  f"test={auc:.4f}   [{time.time()-t0:.0f}s]")

    # ── tabular baselines on the SAME harmonized features ──
    Ztr, Zte, ytr, yte, _ = harmonize_and_scale(df, tr_mask, te_mask)
    for seed in SEEDS:
        models = {
            "dummy": DummyClassifier(strategy="prior"),
            "logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
            "svm_linear": SVC(kernel="linear", probability=True,
                              class_weight="balanced", random_state=seed),
            "svm_rbf": SVC(kernel="rbf", probability=True,
                           class_weight="balanced", random_state=seed),
            "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=seed, n_jobs=-1),
            "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                 random_state=seed),
        }
        if HAS_XGB:
            models["xgboost"] = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                eval_metric="logloss", random_state=seed, n_jobs=-1)
        for name, m in models.items():
            m.fit(Ztr, ytr)
            p = m.predict_proba(Zte)[:, 1]
            rows.append({"model": name, "seed": seed,
                         "test_auc": roc_auc_score(yte, p), "val_auc": np.nan})
        # deterministic models: one seed is enough
        if seed == SEEDS[0]:
            det = [r for r in rows if r["model"] in ("dummy", "logreg", "svm_linear",
                                                     "svm_rbf")]

    res = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"pilot_adaptive_{args.site}.csv")
    res.to_csv(out, index=False)

    summ = (res.groupby("model")["test_auc"].agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False))
    print(f"\n=== holdout {args.site} (n={te_mask.sum()}) — test AUC ===")
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
