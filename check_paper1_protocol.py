"""
How much of Paper 1's 0.635 is architecture, and how much is protocol?

Paper 1's GAT v3 train_one_seed() has no validation set. It evaluates on the
HELD-OUT SITE every epoch, early-stops on that score, schedules the LR on it,
and returns best_auc -- the maximum test AUC seen over up to 200 epochs. The
reported number is therefore a maximum over ~200 noisy draws, not an
out-of-sample estimate.

This measures the resulting optimism directly. ONE training run per (site, seed),
two numbers read off the same curve:

    honest    test AUC at the epoch chosen by a separate VALIDATION split
    paper1    max test AUC over all epochs   (Paper 1's rule)

The gap is the protocol's inflation, isolated from architecture: identical
model, identical data, identical run. Everything else is held constant.

Run over all 20 sites so the result is comparable to Paper 1's 20-site 0.635.

Two reasons this UNDERSTATES Paper 1's true optimism:
  - 60 epochs here vs Paper 1's 200. More epochs = more draws to maximise over.
  - The inflation is bounded by AUC 1.0, so it compresses at high scores.

Resumable: appends per (site, seed) and skips work already on disk.

Outputs (results/):
    paper1_protocol_raw.csv        one row per (site, seed)
    paper1_protocol_check.csv      per-site means
    paper1_protocol_report.txt     verdict

Run:
    python check_paper1_protocol.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

from hetero_data import build_fold
from hetero_gnn import HeteroGNN

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")
COORDS = os.path.join(ROOT, "features", "node_coords.csv")
RAW = os.path.join(OUT_DIR, "paper1_protocol_raw.csv")

SEEDS = [42, 123, 456]
EPOCHS = 60
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
    """Train once; return (honest_auc, paper1_auc) read off the SAME curve."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl = DataLoader(g_tr, batch_size=32, shuffle=True)
    vl = DataLoader(g_va, batch_size=256)
    el = DataLoader(g_te, batch_size=256)

    ys = np.array([int(g.y) for g in g_tr])
    w = torch.tensor([len(ys) / (2 * max((ys == 0).sum(), 1)),
                      len(ys) / (2 * max((ys == 1).sum(), 1))],
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


def summarize():
    raw = pd.read_csv(RAW)
    res = (raw.groupby("site")[["honest", "paper1"]].mean().reset_index())
    res.to_csv(os.path.join(OUT_DIR, "paper1_protocol_check.csv"), index=False)

    n = (pd.read_csv(FEATURES).query("qc_pass").groupby("SITE_ID").size()
         .rename("n_test"))
    res = res.join(n, on="site")
    res["inflation"] = res["paper1"] - res["honest"]

    d = res["inflation"]
    W, p = wilcoxon(res["paper1"], res["honest"])
    r_size, p_size = pearsonr(res["n_test"], res["inflation"])
    r_honest, _ = pearsonr(res["honest"], res["inflation"])

    L = [
        "Protocol optimism: Paper 1's epoch-selection rule vs an honest one",
        "=" * 72,
        "Identical model, data and training run. The ONLY difference is which",
        "epoch is reported: the one chosen on a validation split (honest), or",
        "the best-scoring epoch on the test site itself (Paper 1's rule).",
        "",
        f"sites: {len(res)} | seeds per site: {len(SEEDS)} | epochs: {EPOCHS}",
        "",
        f"  honest       AUC = {res['honest'].mean():.4f} +/- {res['honest'].std():.4f}",
        f"  Paper 1 rule AUC = {res['paper1'].mean():.4f} +/- {res['paper1'].std():.4f}",
        f"  INFLATION        = {d.mean():+.4f} +/- {d.std():.4f}"
        f"   (Wilcoxon p = {p:.4g})",
        f"  range            = {d.min():+.3f} to {d.max():+.3f}",
        "",
        "Does the inflation depend on how big the test site is?",
        f"  corr(site size, inflation)  r = {r_size:+.3f}  p = {p_size:.3f}",
        f"  corr(honest AUC, inflation) r = {r_honest:+.3f}",
        "",
        f"Paper 1 reported GAT v3 at 0.635 +/- 0.052 over these same 20 sites,",
        f"using this rule. The HeteroGNN under the same rule scores "
        f"{res['paper1'].mean():.4f}.",
        "",
        "This measurement UNDERSTATES Paper 1's optimism: 60 epochs here vs its",
        "200, and more epochs means more draws to take a maximum over.",
        "",
        res[["site", "n_test", "honest", "paper1", "inflation"]]
        .sort_values("inflation", ascending=False)
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    ]
    rep = "\n".join(L)
    with open(os.path.join(OUT_DIR, "paper1_protocol_report.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    coords = pd.read_csv(COORDS)
    sites = sorted(df["SITE_ID"].unique())

    done = set()
    if os.path.exists(RAW):
        prev = pd.read_csv(RAW)
        done = set(zip(prev["site"], prev["seed"]))
        print(f"resuming: {len(done)} (site, seed) runs already on disk")

    print(f"device {device} | sites {len(sites)} | seeds {len(SEEDS)} | "
          f"epochs {EPOCHS}", flush=True)

    t0 = time.time()
    for site in sites:
        if all((site, s) in done for s in SEEDS):
            print(f"[{site}] complete, skipping", flush=True)
            continue

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

        for seed in SEEDS:
            if (site, seed) in done:
                continue
            ts = time.time()
            h, p1 = one_run(g_tr, g_va, g_te, seed, device)
            row = {"site": site, "seed": seed, "honest": h, "paper1": p1,
                   "inflation": p1 - h, "n_test": int(tm.sum()),
                   "secs": round(time.time() - ts, 1)}
            pd.DataFrame([row]).to_csv(
                RAW, mode="a", header=not os.path.exists(RAW), index=False)
            print(f"  {site:10s} seed {seed:4d}  honest={h:.3f}  "
                  f"paper1={p1:.3f}  infl={p1-h:+.3f}  ({row['secs']}s)",
                  flush=True)

    print(f"\ntotal {(time.time()-t0)/60:.1f} min", flush=True)
    summarize()


if __name__ == "__main__":
    main()
