"""
Adaptive HeteroGNN under Paper 1's epoch-selection rule.

The ONLY thing that changes vs the honest adaptive run (train_nested_loso_adaptive.py)
is how the reported epoch is chosen. Everything else is held identical:

    * model      : AdaptiveHeteroGNN (per-subject kNN graph), same as the honest run
    * seeds      : the 5 adaptive seeds [42, 123, 456, 789, 1234]
    * HP         : the adaptive run's honest inner-CV choice per site, loaded from
                   results/nested_loso_adaptive_hp.json (NO re-search -- reusing the
                   cache is exactly "keep the hyperparameter search as the adaptive",
                   and it is legitimate because that search never saw the test site)

For each (site, seed) we train ONE model for a fixed epoch budget with NO early
stopping, recording test AUC and validation AUC at every epoch, then read off:

    honest  = test AUC at the epoch that maximised the VALIDATION AUC
    paper1  = max test AUC over all epochs        (Paper 1's rule: peek at the answer)
    inflation = paper1 - honest

Both come from the same run, so inflation is internally valid. Temperature scaling
is skipped -- it is monotonic, so it cannot change AUC.

Comparison target: check_paper1_protocol.py did this for the GROUP-LEVEL HeteroGNN
(honest 0.5497, paper1 0.6502 over 20 sites, 3 seeds, 60 epochs). Keep --epochs
here equal to that (60) if you want the two models on the same footing.

Run:
    python train_paper1rule_adaptive.py --smoke     # 2 sites, 1 seed, few epochs
    python train_paper1rule_adaptive.py             # full: 20 sites x 5 seeds
    python train_paper1rule_adaptive.py --epochs 150  # match the adaptive regime instead
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from torch_geometric.loader import DataLoader

from hetero_data import build_fold
from pilot_adaptive_graph import AdaptiveHeteroGNN
from train_nested_loso_adaptive import (
    set_seed, run_epoch, SEEDS, TAU_FIXED, FEATURES, COORDS, OUT_DIR,
)

HP_CACHE = os.path.join(OUT_DIR, "nested_loso_adaptive_hp.json")
RES_PATH = os.path.join(OUT_DIR, "paper1rule_adaptive_results.csv")


def train_track(tr_graphs, va_graphs, te_graphs, hp, seed, device, epochs, bs=32):
    """Train `epochs` with NO early stopping; return per-epoch (val, test) AUCs.

    Same optimiser / class weighting / scheduler as train_nested_loso_adaptive's
    train_one -- only the stopping logic is removed so the epoch axis is intact
    for the max-over-epochs rule.
    """
    set_seed(seed)
    tr_loader = DataLoader(tr_graphs, batch_size=bs, shuffle=True)
    va_loader = DataLoader(va_graphs, batch_size=256)
    te_loader = DataLoader(te_graphs, batch_size=256)

    ys = np.array([int(g.y) for g in tr_graphs])
    w = torch.tensor(
        [len(ys) / (2 * max((ys == 0).sum(), 1)),
         len(ys) / (2 * max((ys == 1).sum(), 1))],
        dtype=torch.float32,
    ).to(device)

    model = AdaptiveHeteroGNN(hidden=hp["hidden"], heads=4, layers=hp["layers"],
                              dropout=hp["dropout"], k=hp["k"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=5)

    va_hist, te_hist = [], []
    for _ in range(epochs):
        run_epoch(model, tr_loader, device, opt, w)
        _, va_auc, _, _ = run_epoch(model, va_loader, device)
        _, te_auc, _, _ = run_epoch(model, te_loader, device)
        sched.step(va_auc if not np.isnan(va_auc) else 0.0)
        va_hist.append(va_auc)
        te_hist.append(te_auc)
    return np.array(va_hist), np.array(te_hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=60,
                    help="fixed epoch budget (60 matches the group-level paper1 check)")
    ap.add_argument("--no-combat", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if not os.path.exists(HP_CACHE):
        raise SystemExit(
            f"missing {HP_CACHE}\nRun train_nested_loso_adaptive.py first (or restore "
            "the committed cache) so the honest per-site HP choices are available.")
    hp_cache = json.load(open(HP_CACHE))

    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    coords = pd.read_csv(COORDS)
    sites = sorted(df["SITE_ID"].unique())

    seeds = SEEDS[:1] if args.smoke else SEEDS
    epochs = 12 if args.smoke else args.epochs
    outer_sites = sites[:2] if args.smoke else sites
    print(f"subjects: {len(df)}  sites: {len(sites)}  seeds: {len(seeds)}  epochs: {epochs}")

    done = set()
    if os.path.exists(RES_PATH) and not args.smoke:
        prev = pd.read_csv(RES_PATH)
        done = set(zip(prev["site"], prev["seed"]))
        print(f"resuming: {len(done)} (site, seed) rows already on disk")

    t0 = time.time()
    for site in outer_sites:
        if all((site, s) in done for s in seeds):
            print(f"[{site}] complete, skipping")
            continue
        if site not in hp_cache:
            print(f"[{site}] WARNING: no cached HP, skipping")
            continue
        best_hp = hp_cache[site]["hp"]

        test_mask = (df["SITE_ID"] == site).to_numpy()
        train_mask = ~test_mask
        tr_df = df[train_mask]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        tr_i, va_i = next(sss.split(tr_df, tr_df["label"]))
        m_tr = np.zeros(len(df), bool)
        m_va = np.zeros(len(df), bool)
        m_tr[np.flatnonzero(train_mask)[tr_i]] = True
        m_va[np.flatnonzero(train_mask)[va_i]] = True

        g_tr, g_va, _ = build_fold(df, m_tr, m_va, coords, tau=TAU_FIXED,
                                   use_combat=not args.no_combat)
        _, g_te, info = build_fold(df, m_tr, test_mask, coords, tau=TAU_FIXED,
                                   use_combat=not args.no_combat)
        print(f"\n[{site}] hp={best_hp} n_train={info['n_train']} n_test={info['n_test']}")

        for seed in seeds:
            if (site, seed) in done:
                continue
            ts = time.time()
            va_hist, te_hist = train_track(g_tr, g_va, g_te, best_hp, seed,
                                           device, epochs)
            valid = ~np.isnan(va_hist)
            best_ep = int(np.flatnonzero(valid)[np.nanargmax(va_hist[valid])]) \
                if valid.any() else int(np.nanargmax(te_hist))
            honest = float(te_hist[best_ep])
            paper1 = float(np.nanmax(te_hist))
            row = {"site": site, "seed": seed, "honest": round(honest, 4),
                   "paper1": round(paper1, 4), "inflation": round(paper1 - honest, 4),
                   "best_val_epoch": best_ep, "n_test": info["n_test"],
                   "hp": json.dumps(best_hp), "secs": round(time.time() - ts, 1)}
            pd.DataFrame([row]).to_csv(
                RES_PATH, mode="a", header=not os.path.exists(RES_PATH), index=False)
            print(f"  seed {seed:4d}  honest={honest:.3f}  paper1={paper1:.3f}  "
                  f"inflation={paper1-honest:+.3f}  (ep {best_ep}, {row['secs']}s)")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\ntotal {(time.time()-t0)/60:.1f} min -> {RES_PATH}")
    if os.path.exists(RES_PATH):
        r = pd.read_csv(RES_PATH)
        h = r.groupby("site")["honest"].mean()
        p = r.groupby("site")["paper1"].mean()
        print(f"honest       LOSO AUC = {h.mean():.4f} +/- {h.std():.4f}")
        print(f"paper1 rule  LOSO AUC = {p.mean():.4f} +/- {p.std():.4f}")
        print(f"INFLATION             = {(p.mean()-h.mean()):+.4f}  over {len(h)} sites")


if __name__ == "__main__":
    main()
