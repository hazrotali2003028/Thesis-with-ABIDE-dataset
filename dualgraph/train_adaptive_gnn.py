"""
Nested-LOSO training for the adaptive functional GNN (robust4 nodes). Kaggle-ready.

PROTOCOL
  Outer : leave-one-site-out, 17 folds.
  Inner : 3-fold StratifiedGroupKFold(groups=SITE_ID) on the training sites only,
          selecting HP by HONEST validation AUC. The held-out site is never seen
          during selection.
  Seeds : applied after HP selection.

EPOCH SELECTION -- both rules, from the SAME run
  honest : test metrics at the epoch that maximised the train-only VALIDATION AUC.
  paper1 : test metrics at the epoch that maximised the TEST AUC (Paper 1's rule).

  Paper 1's rule is reported for comparison ONLY. Measured on this dataset it
  inflates AUC by +0.100 (p=1.9e-06, see results/paper1_protocol_report.txt); it
  reads the answer key and cannot be the reported result. `inflation` = paper1 -
  honest is logged per site so the gap is explicit. HP selection stays honest under
  both rules -- Paper 1's rule concerns epoch choice only.

STAGED HP SEARCH (compute-bounded; full grid is 48 configs = days)
  stage a : k_adapt in {5,10,20,30}          layers=2, hidden=64   (k is the new tau;
                                                pinning it repeats the tau=0.3 error)
  stage b : layers in {2,3,4} x hidden in {64,128}, at the best k from stage a
            (tests the oversmoothing hypothesis rather than assuming it)

DIAGNOSTIC
  Across-subject SD of the pooled graph vector is logged per config. If depth
  oversmooths, that SD collapses (cf. GraphNorm 0.024 vs BatchNorm 0.199) -- a
  mechanistic signal, unlike a noisy 17-site AUC where the MDE is 0.06.

KAGGLE
  --data-dir /kaggle/input/<dataset>   --out-dir /kaggle/working
  Appends every finished row and caches inner-HP choices, so a 12h session kill is
  recoverable: attach the previous run's notebook output and rerun; done rows are
  skipped.

Run
  python dualgraph/train_adaptive_gnn.py --stage a --smoke
  python dualgraph/train_adaptive_gnn.py --stage a
  python dualgraph/train_adaptive_gnn.py --stage b --best-k 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adaptive_gnn import AdaptiveFuncGNN, make_graph, assert_adaptive_varies  # noqa: E402
from node_features import extract_robust_features                            # noqa: E402

SEEDS = [42, 123, 456, 789, 1234]

# every metric is computed under BOTH epoch-selection rules, prefixed honest_/paper1_
METRIC_KEYS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity",
               "specificity", "precision", "npv", "f1", "mcc"]
COUNT_KEYS = ["tp", "tn", "fp", "fn"]


# ------------------------------------------------------------------ metrics
def compute_metrics(y: np.ndarray, score: np.ndarray, thr: float = 0.5) -> dict:
    """Full publication metric set. Threshold fixed at 0.5 -- selecting it on the
    test site would be another leak."""
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) > 1
    return {
        "roc_auc": roc_auc_score(y, score) if two else np.nan,
        "pr_auc": average_precision_score(y, score) if two else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred) if two else np.nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,   # recall / TPR
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,   # TNR
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,     # PPV
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if two else np.nan,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ train/eval
def _epoch(model, loader, device, opt=None, w=None):
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
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def pooled_sd(model, loader, device) -> float:
    """Across-subject SD of the pooled graph vector (oversmoothing diagnostic)."""
    model.eval()
    gs = []
    for b in loader:
        b = b.to(device)
        gs.append(model(b, return_pooled=True)[1].cpu())
    return float(torch.cat(gs).std(dim=0).mean())


def train_track(graphs, tr_idx, va_idx, te_idx, y, hp, seed, device, epochs):
    """Train a fixed epoch budget (no early stop) tracking val AND test scores,
    so both epoch-selection rules can be read off one run."""
    set_seed(seed)
    tr = [graphs[i] for i in tr_idx]
    ytr = y[tr_idx]
    w = torch.tensor([len(ytr) / (2 * max((ytr == 0).sum(), 1)),
                      len(ytr) / (2 * max((ytr == 1).sum(), 1))],
                     dtype=torch.float32).to(device)
    model = AdaptiveFuncGNN(in_dim=4, hidden=hp["hidden"], heads=4,
                            layers=hp["layers"], dropout=hp["dropout"],
                            k_adapt=hp["k"], variant=hp["variant"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tl = DataLoader(tr, batch_size=16, shuffle=True)
    vl = DataLoader([graphs[i] for i in va_idx], batch_size=64)
    el = DataLoader([graphs[i] for i in te_idx], batch_size=64) if te_idx is not None else None

    va_hist, te_scores = [], []
    for _ in range(epochs):
        _epoch(model, tl, device, opt, w)
        pv, yv = _epoch(model, vl, device)
        va_hist.append(roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else np.nan)
        if el is not None:
            pt, _ = _epoch(model, el, device)
            te_scores.append(pt)
    return model, np.array(va_hist), (np.stack(te_scores) if te_scores else None), vl


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=HERE,
                    help="dir holding cohort_final.csv and cache/fc_z.npy "
                         "(on Kaggle: /kaggle/input/<dataset>)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"),
                    help="on Kaggle: /kaggle/working")
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    ap.add_argument("--best-k", type=int, default=10, help="stage b: k from stage a")
    ap.add_argument("--variant", choices=["fc", "adapt", "dual"], default="dual")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  stage: {args.stage}  variant: {args.variant}")

    coh = pd.read_csv(os.path.join(args.data_dir, "cohort_final.csv"))
    fc = np.load(os.path.join(args.data_dir, "cache", "fc_z.npy"))
    y = (coh.DX_GROUP == 1).to_numpy(int)
    sites = coh.SITE_ID.to_numpy()
    print(f"cohort {len(coh)}  ASD {y.sum()}  TD {(y==0).sum()}  sites {len(set(sites))}")

    # robust4 node features, recomputed from fc_z (~12s) so nothing extra to upload
    t0 = time.time()
    X = np.stack([extract_robust_features(fc[i]) for i in range(len(coh))])
    print(f"robust4 {X.shape}  ({time.time()-t0:.0f}s)")

    grid = ([{"k": k, "layers": 2, "hidden": 64, "dropout": 0.3, "variant": args.variant}
             for k in (5, 10, 20, 30)] if args.stage == "a" else
            [{"k": args.best_k, "layers": L, "hidden": H, "dropout": 0.3,
              "variant": args.variant}
             for L, H in itertools.product((2, 3, 4), (64, 128))])
    seeds = SEEDS[:args.seeds]
    usites = sorted(np.unique(sites))
    epochs = args.epochs
    if args.smoke:
        grid, seeds, usites, epochs = grid[:1], seeds[:1], usites[:2], 6
    print(f"grid {len(grid)} configs  seeds {len(seeds)}  sites {len(usites)}  epochs {epochs}")

    res_path = os.path.join(args.out_dir, f"adaptive_gnn_stage{args.stage}.csv")
    hp_path = os.path.join(args.out_dir, f"adaptive_gnn_stage{args.stage}_hp.json")
    done = set()
    if os.path.exists(res_path) and not args.smoke:
        prev = pd.read_csv(res_path)
        done = set(zip(prev.site, prev.seed))
        print(f"resuming: {len(done)} (site,seed) rows already done")
    hp_cache = json.load(open(hp_path)) if os.path.exists(hp_path) and not args.smoke else {}

    # graphs are rebuilt per k (fc topology depends on k)
    graph_cache: dict[int, list] = {}

    def graphs_for(k):
        if k not in graph_cache:
            graph_cache[k] = [make_graph(X[i], fc[i], int(y[i]), k=k)
                              for i in range(len(coh))]
        return graph_cache[k]

    checked_adaptive = False
    t_start = time.time()
    for site in usites:
        if all((site, s) in done for s in seeds):
            print(f"[{site}] complete, skipping"); continue
        te_idx = np.flatnonzero(sites == site)
        tr_all = np.flatnonzero(sites != site)

        # ---- INNER: honest HP selection on training sites only ----
        if site in hp_cache:
            best = hp_cache[site]["hp"]
            print(f"\n[{site}] cached hp={best}")
        else:
            sgkf = StratifiedGroupKFold(n_splits=args.inner_folds)
            scores = {json.dumps(h): [] for h in grid}
            for itr, iva in sgkf.split(tr_all, y[tr_all], groups=sites[tr_all]):
                a_tr, a_va = tr_all[itr], tr_all[iva]
                for h in grid:
                    _, vh, _, _ = train_track(graphs_for(h["k"]), a_tr, a_va, None,
                                              y, h, seeds[0], device, epochs)
                    scores[json.dumps(h)].append(np.nanmax(vh))
            best = max(grid, key=lambda h: float(np.nanmean(scores[json.dumps(h)])))
            print(f"\n[{site}] inner-best hp={best}")
            if not args.smoke:
                hp_cache[site] = {"hp": best}
                json.dump(hp_cache, open(hp_path, "w"), indent=2)

        G = graphs_for(best["k"])
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=0)
        rtr, rva = next(sss.split(tr_all, y[tr_all]))
        tr_idx, va_idx = tr_all[rtr], tr_all[rva]

        for seed in seeds:
            if (site, seed) in done:
                continue
            ts = time.time()
            model, vh, te_hist, vl = train_track(G, tr_idx, va_idx, te_idx, y,
                                                 best, seed, device, epochs)
            if not checked_adaptive and best["variant"] in ("adapt", "dual"):
                sd = assert_adaptive_varies(model, vl, device)
                print(f"  [unit test] adaptive edge_index std across subjects = {sd:.4f} (>0 OK)")
                checked_adaptive = True

            yte = y[te_idx]
            ep_honest = int(np.nanargmax(vh))
            aucs = [roc_auc_score(yte, te_hist[e]) if len(np.unique(yte)) > 1 else np.nan
                    for e in range(te_hist.shape[0])]
            ep_paper1 = int(np.nanargmax(aucs))

            row = {"stage": args.stage, "site": site, "seed": seed,
                   "n_test": len(te_idx), "hp": json.dumps(best),
                   "epoch_honest": ep_honest, "epoch_paper1": ep_paper1,
                   "pooled_sd": round(pooled_sd(model, vl, device), 4),
                   "secs": round(time.time() - ts, 1)}
            for tag, ep in (("honest", ep_honest), ("paper1", ep_paper1)):
                for kk, vv in compute_metrics(yte, te_hist[ep]).items():
                    row[f"{tag}_{kk}"] = round(vv, 4) if isinstance(vv, float) else vv
            row["inflation"] = round(row["paper1_roc_auc"] - row["honest_roc_auc"], 4)

            pd.DataFrame([row]).to_csv(res_path, mode="a",
                                       header=not os.path.exists(res_path), index=False)
            print(f"  seed {seed:4d} honest={row['honest_roc_auc']:.3f} "
                  f"paper1={row['paper1_roc_auc']:.3f} (infl {row['inflation']:+.3f})  "
                  f"acc={row['honest_accuracy']:.3f} f1={row['honest_f1']:.3f} "
                  f"spec={row['honest_specificity']:.3f}  pooledSD={row['pooled_sd']:.3f} "
                  f"({row['secs']}s)")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\ntotal {(time.time()-t_start)/60:.1f} min -> {res_path}")
    if os.path.exists(res_path):
        report(pd.read_csv(res_path))


def report(r: pd.DataFrame):
    """Per-site tables under BOTH epoch rules, plus the per-metric inflation."""
    per = r.groupby("site").mean(numeric_only=True)

    for tag, note in (("honest", "REPORT THIS"),
                      ("paper1", "COMPARISON ONLY -- epoch chosen on the test site")):
        cols = [f"{tag}_{m}" for m in METRIC_KEYS if f"{tag}_{m}" in per]
        print(f"\n=== per-site mean, {tag.upper()} rule ({note}) ===")
        print(per[cols].rename(columns=lambda c: c.replace(f"{tag}_", ""))
              .round(3).to_string())

    print("\n=== inflation per metric (paper1 - honest, mean over sites) ===")
    rows = []
    for m in METRIC_KEYS:
        h, p = f"honest_{m}", f"paper1_{m}"
        if h in per and p in per:
            rows.append({"metric": m, "honest": per[h].mean(),
                         "paper1": per[p].mean(),
                         "inflation": per[p].mean() - per[h].mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print(f"\nHONEST LOSO ROC = {per.honest_roc_auc.mean():.4f} "
          f"+/- {per.honest_roc_auc.std():.4f}   <- the reportable number")
    print(f"paper1-rule ROC = {per.paper1_roc_auc.mean():.4f} "
          f"(+{per.inflation.mean():.4f}) -- NOT reportable; it reads the answer key")
    print(f"pooled-vector SD (oversmoothing diagnostic) = {per.pooled_sd.mean():.4f}")
    print("anchors: edge-SVM 0b=0.658  robust4 flat SVM=0.630  motion floor=0.561")


if __name__ == "__main__":
    main()
