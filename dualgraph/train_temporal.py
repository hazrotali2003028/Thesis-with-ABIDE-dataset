"""
Train the temporal-encoder functional GNN under LOSO, with the G1 motion gate.

Protocol: leave-one-site-out (17 folds). Within each fold a stratified train/val
split (train sites only) drives early stopping; the held-out site is predicted
once, giving out-of-fold P(ASD) for every subject. Fixed HP (this is the
feasibility build; nested HP search can wrap it later). Class-weighted CE.

Reports vs the two anchors already established:
  * edge-SVM ceiling  0b = 0.658  (does the temporal signal beat static FC?)
  * motion floor      -2 = 0.561
And runs G1 on the out-of-fold predictions -- the decisive check that a temporal
model (which sees raw BOLD) is not a motion detector:
  r(score, FD), r(score, label), partial r(label, score | FD), per-site coupling.
If the temporal GNN wins on AUC but fails G1, that is not a real win.

Run:  python dualgraph/train_temporal.py --sites 2 --epochs 15   # smoke
      python dualgraph/train_temporal.py                          # full 17-fold
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from temporal_gnn import make_graph, TemporalFuncGNN            # noqa: E402
from motion_gates import partial_r                             # noqa: E402

COHORT = os.path.join(HERE, "cohort_final.csv")
FC = os.path.join(HERE, "cache", "fc_z.npy")
TS = os.path.join(HERE, "cache", "ts.npy")
OUTDIR = os.path.join(HERE, "results")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run(model, loader, device, opt=None, w=None):
    train = opt is not None
    model.train(train)
    ps, ys = [], []
    for b in loader:
        b = b.to(device)
        with torch.set_grad_enabled(train):
            logit = model(b)
            loss = F.cross_entropy(logit, b.y, weight=w)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ps.append(F.softmax(logit, 1)[:, 1].detach().cpu()); ys.append(b.y.cpu())
    p = torch.cat(ps).numpy(); y = torch.cat(ys).numpy()
    return (roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan), p, y


def train_fold(graphs, tr_idx, va_idx, y, device, epochs, patience, seed, k_hp):
    set_seed(seed)
    tr = [graphs[i] for i in tr_idx]; va = [graphs[i] for i in va_idx]
    ytr = y[tr_idx]
    w = torch.tensor([len(ytr) / (2 * max((ytr == 0).sum(), 1)),
                      len(ytr) / (2 * max((ytr == 1).sum(), 1))],
                     dtype=torch.float32).to(device)
    model = TemporalFuncGNN(**k_hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = DataLoader(tr, batch_size=16, shuffle=True)
    vl = DataLoader(va, batch_size=64)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        run(model, tl, device, opt, w)
        va_auc, _, _ = run(model, vl, device)
        if va_auc > best:
            best, bad = va_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    coh = pd.read_csv(COHORT)
    fc = np.load(FC); ts = np.load(TS)
    y = (coh.DX_GROUP == 1).to_numpy(int)
    fd = coh.func_mean_fd.to_numpy(float)
    sites = coh.SITE_ID.to_numpy()
    hp = dict(enc_dim=16, hidden=64, heads=4, layers=2, dropout=0.3)

    print("building graphs...")
    graphs = [make_graph(ts[i], fc[i], int(y[i]), k=args.k) for i in range(len(coh))]

    usites = sorted(np.unique(sites))[:args.sites] if args.sites else sorted(np.unique(sites))
    oof = np.full(len(y), np.nan)
    rows = []
    for s in usites:
        te = np.flatnonzero(sites == s)
        tr_all = np.flatnonzero(sites != s)
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=0)
        rel_tr, rel_va = next(sss.split(tr_all, y[tr_all]))
        tr_idx, va_idx = tr_all[rel_tr], tr_all[rel_va]
        model = train_fold(graphs, tr_idx, va_idx, y, device,
                           args.epochs, args.patience, args.seed, hp)
        tl = DataLoader([graphs[i] for i in te], batch_size=64)
        auc, p, yy = run(model, tl, device)
        oof[te] = p
        pr = average_precision_score(yy, p)
        rows.append({"site": s, "n_test": len(te), "roc_auc": round(auc, 4),
                     "pr_auc": round(pr, 4)})
        print(f"  {s:9} roc={auc:.3f} pr={pr:.3f} n={len(te)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "temporal_gnn_results.csv"), index=False)
    print(f"\ntemporal-GNN LOSO ROC {df.roc_auc.mean():.4f} +/- {df.roc_auc.std():.4f}"
          f"   PR {df.pr_auc.mean():.4f}")
    print("anchors: edge-SVM 0b=0.658   motion floor -2=0.561")

    # ---- G1 motion gate on out-of-fold predictions ----
    m = ~np.isnan(oof)
    print("\n=== G1 motion gate (temporal model sees raw BOLD) ===")
    if m.sum() > 10 and len(np.unique(y[m])) > 1:
        print(f"  r(score, FD)          = {pearsonr(oof[m], fd[m])[0]:+.3f} "
              f"(p={pearsonr(oof[m], fd[m])[1]:.3g})")
        print(f"  r(score, label)       = {pointbiserialr(y[m], oof[m])[0]:+.3f}")
        pr_, pp_ = partial_r(y[m].astype(float), oof[m], fd[m])
        print(f"  partial r(label|FD)   = {pr_:+.3f} (p={pp_:.3g})   signal after FD control")
        print("  reference: edge-model r(score,FD) was +0.137 (not a motion detector)")
        gate = "PASS: tracks label > motion, survives FD control" \
            if abs(pointbiserialr(y[m], oof[m])[0]) > abs(pearsonr(oof[m], fd[m])[0]) and pp_ < 0.05 \
            else "REVIEW: motion coupling not clearly beaten"
        print(f"  VERDICT: {gate}")
    np.save(os.path.join(OUTDIR, "temporal_gnn_oof.npy"), oof)
    print(f"\nwrote {os.path.join(OUTDIR,'temporal_gnn_results.csv')}")


if __name__ == "__main__":
    main()
