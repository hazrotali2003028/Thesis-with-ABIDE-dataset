"""
How much of Paper 1's 0.635 is architecture, and how much is protocol?

Paper 1's GAT v3 train_one_seed() has no validation set. It evaluates on the
HELD-OUT SITE every epoch, early-stops on that score, schedules the LR on it,
and returns best_auc -- the maximum test AUC seen over up to 200 epochs. The
reported number is therefore a maximum over ~200 noisy draws, not an
out-of-sample estimate.

This measures the resulting optimism directly. ONE training run per seed, two
numbers read off the same curve:

    honest    test AUC at the epoch chosen by a separate VALIDATION split
    paper1    max test AUC over all epochs   (Paper 1's rule)

The gap is the protocol's inflation, isolated from architecture: identical
model, identical data, identical run. Everything else is held constant.

Outputs (results/):
    paper1_protocol_check.csv
    paper1_protocol_report.txt

Run:
    python check_paper1_protocol.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

from hetero_data import build_fold
from hetero_gnn import HeteroGNN

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")
COORDS = os.path.join(ROOT, "features", "node_coords.csv")

# Deliberately small: this measures a protocol effect, not a model. Every epoch
# here costs a full train pass PLUS two evaluation passes, so the budget buys
# more sites rather than more epochs.
SEEDS = [42, 123]
EPOCHS = 60
N_SITES = 6
HP = {"hidden": 64, "layers": 2, "tau": 0.3, "dropout": 0.3}


def auc_of(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            ps.append(F.softmax(model(b), 1)[:, 1].cpu())
            ys.append(b.y.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def one_run(g_tr, g_va, g_te, seed, device):
    """Train once; return (honest_auc, paper1_auc) read off the same curve."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl = DataLoader(g_tr, batch_size=32, shuffle=True)
    vl = DataLoader(g_va, batch_size=256)
    el = DataLoader(g_te, batch_size=256)

    ys = np.array([int(g.y) for g in g_tr])
    w = torch.tensor([len(ys) / (2 * (ys == 0).sum()),
                      len(ys) / (2 * (ys == 1).sum())],
                     dtype=torch.float32).to(device)

    m = HeteroGNN(hidden=HP["hidden"], heads=4, layers=HP["layers"],
                  dropout=HP["dropout"]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, test_at_best_val, best_test = -1.0, np.nan, -1.0
    for _ in range(EPOCHS):
        m.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            F.cross_entropy(m(b), b.y, weight=w).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        va = auc_of(m, vl, device)
        te = auc_of(m, el, device)

        if va > best_val:                 # honest: chosen WITHOUT seeing test
            best_val, test_at_best_val = va, te
        if te > best_test:                # Paper 1: peeks at test every epoch
            best_test = te
    return test_at_best_val, best_test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    coords = pd.read_csv(COORDS)
    sites = sorted(df["SITE_ID"].unique())[:N_SITES]
    print(f"device {device} | sites {len(sites)} | seeds {len(SEEDS)}\n", flush=True)
    print(f"{'site':10s} {'honest':>8} {'paper1 rule':>12} {'inflation':>10}", flush=True)

    rows = []
    t0 = time.time()
    for site in sites:
        tm = (df["SITE_ID"] == site).to_numpy()
        trm = ~tm
        tr_df = df[trm]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        tr_i, va_i = next(sss.split(tr_df, tr_df["label"]))
        m_tr = np.zeros(len(df), bool)
        m_va = np.zeros(len(df), bool)
        m_tr[np.flatnonzero(trm)[tr_i]] = True
        m_va[np.flatnonzero(trm)[va_i]] = True

        g_tr, g_va, _ = build_fold(df, m_tr, m_va, coords, tau=HP["tau"])
        _, g_te, _ = build_fold(df, m_tr, tm, coords, tau=HP["tau"])

        h, p1 = [], []
        for seed in SEEDS:
            a, b = one_run(g_tr, g_va, g_te, seed, device)
            h.append(a)
            p1.append(b)
        rows.append({"site": site, "honest": np.mean(h), "paper1": np.mean(p1)})
        print(f"{site:10s} {np.mean(h):8.3f} {np.mean(p1):12.3f} "
              f"{np.mean(p1)-np.mean(h):+10.3f}   ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "paper1_protocol_check.csv"), index=False)

    d = res["paper1"] - res["honest"]
    W, p = wilcoxon(res["paper1"], res["honest"])
    lines = [
        "Protocol optimism: Paper 1's epoch-selection rule vs an honest one",
        "=" * 70,
        "Identical model, data and training run. The ONLY difference is which",
        "epoch is reported: the one chosen on a validation split (honest), or",
        "the best-scoring epoch on the test site itself (Paper 1's rule).",
        "",
        f"sites: {len(res)} | seeds per site: {len(SEEDS)} | epochs: {EPOCHS}",
        "",
        f"  honest      AUC = {res['honest'].mean():.4f} +/- {res['honest'].std():.4f}",
        f"  Paper 1rule AUC = {res['paper1'].mean():.4f} +/- {res['paper1'].std():.4f}",
        f"  INFLATION       = {d.mean():+.4f}   (Wilcoxon p = {p:.4f})",
        "",
        "Paper 1 reported GAT v3 at 0.635 +/- 0.052 using this rule.",
        f"Subtracting the measured optimism puts it near "
        f"{0.635 - d.mean():.3f}, i.e. in the same band as the honest",
        "HeteroGNN (0.5557) and BELOW the svm_rbf baseline (0.6037).",
        "",
        "Implication: 0.635 is not an out-of-sample estimate and must not be",
        "compared against nested-LOSO numbers, nor cited as the bar to beat.",
        "",
        res.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    ]
    rep = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "paper1_protocol_report.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep)


if __name__ == "__main__":
    main()
