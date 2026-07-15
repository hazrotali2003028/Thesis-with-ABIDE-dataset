"""
Nested leave-one-site-out training (plan Stages 5 and 8).

Ordering is the leakage guarantee, so it is enforced structurally here:

    for each held-out site S:                    # OUTER: 20 folds
        inner CV over the remaining 19 sites     # hyperparameters chosen here
            -> best config                       #   S is never touched
        refit on all training sites, early-stop
        and temperature-scale on a validation
        split carved from TRAINING sites only
        -> evaluate once on S

ComBat, the scaler and the covariation adjacency are all fitted inside
build_fold() on the training rows of the current split, never on S.

Run:
    python train_nested_loso.py --smoke     # 2 folds, 1 config, 1 seed
    python train_nested_loso.py             # full run (hours; resumable)
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

from hetero_data import build_fold
from hetero_gnn import HeteroGNN, fit_temperature, expected_calibration_error

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")
COORDS = os.path.join(ROOT, "features", "node_coords.csv")

SEEDS = [42, 123, 456, 789, 1234]
HP_GRID = {"hidden": [32, 64], "layers": [2], "tau": [0.3], "dropout": [0.3]}


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, device, opt=None, class_w=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
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
        total += float(loss) * b.num_graphs
        n += b.num_graphs
        probs.append(F.softmax(logits, 1)[:, 1].detach().cpu())
        ys.append(b.y.detach().cpu())
    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()
    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    return total / n, auc, probs, ys


def train_one(tr_graphs, va_graphs, hp, seed, device, epochs, patience, bs=32):
    set_seed(seed)
    tr_loader = DataLoader(tr_graphs, batch_size=bs, shuffle=True)
    va_loader = DataLoader(va_graphs, batch_size=256)

    # Class weights from THIS fold's training composition, not global counts:
    # per-site ASD rate ranges from ~39% to ~65% across ABIDE sites.
    ys = np.array([int(g.y) for g in tr_graphs])
    w = torch.tensor(
        [len(ys) / (2 * max((ys == 0).sum(), 1)),
         len(ys) / (2 * max((ys == 1).sum(), 1))],
        dtype=torch.float32,
    ).to(device)

    model = HeteroGNN(hidden=hp["hidden"], heads=4,
                      layers=hp["layers"], dropout=hp["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=5)

    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        run_epoch(model, tr_loader, device, opt, w)
        _, va_auc, _, _ = run_epoch(model, va_loader, device)
        sched.step(va_auc if not np.isnan(va_auc) else 0.0)
        if va_auc > best_auc:
            best_auc, bad = va_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--no-combat", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    coords = pd.read_csv(COORDS)
    sites = sorted(df["SITE_ID"].unique())
    print(f"subjects: {len(df)}  sites: {len(sites)}")

    seeds = SEEDS[:1] if args.smoke else SEEDS
    epochs = 12 if args.smoke else args.epochs
    patience = 5 if args.smoke else args.patience
    outer_sites = sites[:2] if args.smoke else sites
    grid = [dict(zip(HP_GRID, v)) for v in itertools.product(*HP_GRID.values())]
    if args.smoke:
        grid = grid[:1]

    res_path = os.path.join(OUT_DIR, "nested_loso_results.csv")
    done = set()
    if os.path.exists(res_path) and not args.smoke:
        prev = pd.read_csv(res_path)
        done = set(zip(prev["site"], prev["seed"]))
        print(f"resuming: {len(done)} (site, seed) results already on disk")

    t0 = time.time()
    for site in outer_sites:
        # Skip the whole site when every seed is already on disk: the inner HP
        # search is the expensive part and re-running it would waste the resume.
        if all((site, s) in done for s in seeds):
            print(f"[{site}] complete, skipping")
            continue

        test_mask = (df["SITE_ID"] == site).to_numpy()
        train_mask = ~test_mask
        tr_df = df[train_mask]

        # ── INNER: hyperparameter selection over TRAINING sites only ──
        best_hp, best_score = grid[0], -1.0
        if len(grid) > 1:
            groups = tr_df["SITE_ID"].to_numpy()
            gkf = GroupKFold(n_splits=args.inner_folds)
            for hp in grid:
                scores = []
                for itr, iva in gkf.split(tr_df, groups=groups):
                    m_tr = np.zeros(len(df), bool)
                    m_va = np.zeros(len(df), bool)
                    m_tr[np.flatnonzero(train_mask)[itr]] = True
                    m_va[np.flatnonzero(train_mask)[iva]] = True
                    g_tr, g_va, _ = build_fold(df, m_tr, m_va, coords,
                                               tau=hp["tau"],
                                               use_combat=not args.no_combat)
                    _, auc = train_one(g_tr, g_va, hp, SEEDS[0], device,
                                       epochs, patience)
                    scores.append(auc)
                s = float(np.nanmean(scores))
                if s > best_score:
                    best_score, best_hp = s, hp
        print(f"\n[{site}] inner-best hp={best_hp} inner_auc={best_score:.3f}")

        # ── OUTER: refit on all training sites, evaluate once on the held-out site ──
        # Validation for early stopping / temperature comes from TRAINING sites.
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        tr_i, va_i = next(sss.split(tr_df, tr_df["label"]))
        m_tr = np.zeros(len(df), bool)
        m_va = np.zeros(len(df), bool)
        m_tr[np.flatnonzero(train_mask)[tr_i]] = True
        m_va[np.flatnonzero(train_mask)[va_i]] = True

        g_tr, g_va, _ = build_fold(df, m_tr, m_va, coords, tau=best_hp["tau"],
                                   use_combat=not args.no_combat)
        _, g_te, info = build_fold(df, m_tr, test_mask, coords, tau=best_hp["tau"],
                                   use_combat=not args.no_combat)
        print(f"[{site}] n_train={info['n_train']} n_test={info['n_test']} "
              f"edges cc/ss/cs={info['edges_cc']}/{info['edges_ss']}/{info['edges_cs']}")

        for seed in seeds:
            if (site, seed) in done:
                continue
            ts = time.time()
            model, va_auc = train_one(g_tr, g_va, best_hp, seed, device, epochs, patience)
            temp = fit_temperature(model, DataLoader(g_va, batch_size=256), device)

            te_loader = DataLoader(g_te, batch_size=256)
            model.eval()
            probs, ys = [], []
            with torch.no_grad():
                for b in te_loader:
                    b = b.to(device)
                    probs.append(model.predict_proba(b).cpu())
                    ys.append(b.y.cpu())
            probs = torch.cat(probs).numpy()
            ys = torch.cat(ys).numpy()
            auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
            ece = expected_calibration_error(probs, ys)

            row = {"site": site, "seed": seed, "test_auc": auc, "val_auc": va_auc,
                   "ece": ece, "temperature": temp, "n_test": len(ys),
                   "hp": json.dumps(best_hp), "secs": round(time.time() - ts, 1)}
            pd.DataFrame([row]).to_csv(
                res_path, mode="a", header=not os.path.exists(res_path), index=False)
            print(f"  seed {seed:4d}  test_auc={auc:.3f}  ece={ece:.3f}  "
                  f"T={temp:.2f}  ({row['secs']}s)")

    el = (time.time() - t0) / 60
    print(f"\ntotal {el:.1f} min -> {res_path}")
    if os.path.exists(res_path):
        r = pd.read_csv(res_path)
        per_site = r.groupby("site")["test_auc"].mean()
        print(f"mean LOSO AUC = {per_site.mean():.4f} +/- {per_site.std():.4f} "
              f"over {len(per_site)} sites")


if __name__ == "__main__":
    main()
