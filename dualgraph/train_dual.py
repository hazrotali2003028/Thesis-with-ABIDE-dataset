"""
Stages 7-8 runner: rungs 1a, 1b, 1a-mlp, 1b-mlp, 2 (fusion), 3/4 (cross-attention).

PROTOCOL (identical to the rest of the ladder)
  Outer : leave-one-site-out, 17 folds; test site predicted once.
  Val   : 85/15 stratified split of the TRAINING sites, random_state=0.
  Seeds : 5 -- [42,123,456,789,1234], applied after HP is fixed.
  Epochs: 60, NO early stopping, so both epoch rules come from one run.

TWO VALIDATION RESULTS from the same run
  honest : test metrics at the epoch maximising the train-only VALIDATION AUC.
  paper1 : test metrics at the epoch maximising the TEST AUC (Paper 1's rule).
  Paper 1's rule is comparison-only: measured +0.100 inflation (p=1.9e-06) on this
  data, and it distorts the operating point (it optimises a ranking metric while the
  threshold stays at 0.5). Full 14-metric set under BOTH rules, plus per-metric
  inflation. Never report the paper1 columns as a result.

HARMONISATION -- fixes a real confound
  The flat SVM baselines (0a/0b/robust4) used PER-FOLD ComBat; the earlier GNN runs
  used only a global z-score, i.e. no site harmonisation. Under LOSO that is not a
  fair comparison. Here BOTH arms get per-fold ComBat fitted on training rows only:
    functional : ComBat on the 6105 upper-triangle Fisher-z edges, then the matrix is
                 reconstructed -- serving as BOTH the fcrow node features and the
                 edge weights.
    structural : ComBat on the 232-vector via harmonize_and_scale.
  Protected covariates [DX, age, sex, FD]; dx is dropped on the unseen site.

CHECKPOINTING
  One row appended per (site, seed) the moment it finishes -> a killed session loses
  at most one fold. Resume skips finished (site, seed) pairs.

Run
  python dualgraph/train_dual.py --rung 1a --smoke
  python dualgraph/train_dual.py --rung 2 --data-dir /kaggle/working/data --out-dir /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from combat import combat_fit, combat_apply_train, combat_apply_unseen   # noqa: E402
from hetero_data import (CORTICAL_NODES, SUBCORTICAL_NODES, CORTICAL_MEASURES,
                         feature_matrix, harmonize_and_scale)             # noqa: E402
from dual_graph import build_model, N_FUNC, N_CORT, N_SUB                 # noqa: E402
from train_adaptive_gnn import compute_metrics, METRIC_KEYS, set_seed, report  # noqa: E402

SEEDS = [42, 123, 456, 789, 1234]
RUNGS = ["1a", "1b", "1a-mlp", "1b-mlp", "2", "3", "4"]


# ------------------------------------------------------------------ data
def load_all(data_dir):
    coh = pd.read_csv(os.path.join(data_dir, "cohort_final.csv"))
    fc = np.load(os.path.join(data_dir, "cache", "fc_z.npy")).astype(np.float64)
    feat = pd.read_csv(os.path.join(data_dir, "features", "abide_features_raw.csv")) \
        if os.path.exists(os.path.join(data_dir, "features", "abide_features_raw.csv")) \
        else pd.read_csv(os.path.join(ROOT, "features", "abide_features_raw.csv"))
    coords = pd.read_csv(os.path.join(ROOT, "features", "node_coords.csv")) \
        if os.path.exists(os.path.join(ROOT, "features", "node_coords.csv")) \
        else pd.read_csv(os.path.join(data_dir, "features", "node_coords.csv"))
    sdf = coh[["SUB_ID"]].merge(feat, on="SUB_ID", how="left").reset_index(drop=True)
    y = (coh.DX_GROUP == 1).to_numpy(int)
    sdf["label"] = y
    return coh, fc, sdf, coords, y


def coord_tensors(coords):
    cm = coords.set_index("node")[["x", "y", "z"]]
    P = np.vstack([cm.loc[CORTICAL_NODES].to_numpy(float),
                   cm.loc[SUBCORTICAL_NODES].to_numpy(float)])
    P = (P - P.mean(0)) / P.std(0)                     # group constants (positional)
    return P[:N_CORT], P[N_CORT:]


def prep_fold(coh, fc, sdf, Pc, Ps, tr, te, use_combat=True):
    """Per-fold ComBat on BOTH modalities, fitted on training rows only."""
    iu = np.triu_indices(N_FUNC, 1)
    E = fc[:, iu[0], iu[1]]                                            # [N,6105]
    dx = (coh.DX_GROUP == 1).to_numpy(float)
    Xp = np.column_stack([dx, coh.AGE_AT_SCAN.to_numpy(float),
                          coh.SEX.to_numpy(float), coh.func_mean_fd.to_numpy(float)])
    site = coh.SITE_ID.to_numpy()

    if use_combat:
        est = combat_fit(E[tr], site[tr], Xp[tr], [1, 2, 3])
        Etr = combat_apply_train(E[tr], site[tr], Xp[tr], est)
        Ete = combat_apply_unseen(E[te], Xp[te], est)
    else:
        Etr, Ete = E[tr], E[te]
    mu, sd = Etr.mean(0), Etr.std(0); sd[sd == 0] = 1.0
    Eh = np.zeros_like(E)
    Eh[tr], Eh[te] = (Etr - mu) / sd, (Ete - mu) / sd
    FCh = np.zeros_like(fc)                                            # rebuild matrix
    FCh[:, iu[0], iu[1]] = Eh
    FCh = FCh + np.transpose(FCh, (0, 2, 1))

    Ztr, Zte, *_ = harmonize_and_scale(sdf, tr, te, use_combat=use_combat)
    Z = np.zeros((len(sdf), Ztr.shape[1])); Z[tr], Z[te] = Ztr, Zte
    n_sub = len(SUBCORTICAL_NODES)
    vol = Z[:, :n_sub][:, :, None]                                     # [N,28,1]
    cort = Z[:, n_sub:].reshape(len(Z), len(CORTICAL_NODES), len(CORTICAL_MEASURES))
    x_c = np.concatenate([cort, np.broadcast_to(Pc, (len(Z),) + Pc.shape)], axis=2)
    x_s = np.concatenate([vol, np.broadcast_to(Ps, (len(Z),) + Ps.shape)], axis=2)

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {"x_f": t(FCh), "fc": t(FCh), "x_c": t(x_c), "x_s": t(x_s)}


def batches(T, idx, bs, shuffle, device):
    order = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        yield {k: v[j].to(device) for k, v in T.items()}, j


# ------------------------------------------------------------------ train
def run_fold(T, tr, va, te, y, rung, hp, seed, device, epochs, bs=16):
    set_seed(seed)
    model = build_model(rung, hidden=hp["hidden"], layers=hp["layers"],
                        dropout=hp["dropout"], k=hp["k"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ytr = y[tr]
    w = torch.tensor([len(ytr) / (2 * max((ytr == 0).sum(), 1)),
                      len(ytr) / (2 * max((ytr == 1).sum(), 1))],
                     dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long)

    from sklearn.metrics import roc_auc_score
    va_hist, te_hist = [], []
    for _ in range(epochs):
        model.train()
        for b, j in batches(T, tr, bs, True, device):
            loss = F.cross_entropy(model(b), yt[j].to(device), weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            sc = {}
            for name, idx in (("va", va), ("te", te)):
                ps = [F.softmax(model(b), 1)[:, 1].cpu() for b, _ in
                      batches(T, idx, 64, False, device)]
                sc[name] = torch.cat(ps).numpy()
        va_hist.append(roc_auc_score(y[va], sc["va"]) if len(np.unique(y[va])) > 1 else np.nan)
        te_hist.append(sc["te"])
    return model, np.array(va_hist), np.stack(te_hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=RUNGS, required=True)
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-combat", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coh, fc, sdf, coords, y = load_all(args.data_dir)
    Pc, Ps = coord_tensors(coords)
    sites = coh.SITE_ID.to_numpy()
    hp = {"k": args.k, "layers": args.layers, "hidden": args.hidden,
          "dropout": args.dropout}
    seeds = SEEDS[:args.seeds]
    usites = sorted(np.unique(sites))
    epochs = args.epochs
    if args.smoke:
        seeds, usites, epochs = seeds[:1], usites[:2], 4
    print(f"device {device} | rung {args.rung} | hp {hp} | seeds {len(seeds)} "
          f"| sites {len(usites)} | epochs {epochs} | combat {not args.no_combat}")

    res = os.path.join(args.out_dir, f"dual_rung{args.rung}.csv")
    done = set()
    if os.path.exists(res) and not args.smoke:
        prev = pd.read_csv(res); done = set(zip(prev.site, prev.seed))
        print(f"resuming: {len(done)} (site,seed) rows done")

    t0 = time.time()
    for s in usites:
        if all((s, sd) in done for sd in seeds):
            print(f"[{s}] complete, skipping"); continue
        te = np.flatnonzero(sites == s)
        tr_all = np.flatnonzero(sites != s)
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=0)
        rtr, rva = next(sss.split(tr_all, y[tr_all]))
        tr, va = tr_all[rtr], tr_all[rva]
        T = prep_fold(coh, fc, sdf, Pc, Ps, tr_all, te, use_combat=not args.no_combat)
        print(f"\n[{s}] n_train={len(tr)} n_val={len(va)} n_test={len(te)}")

        for seed in seeds:
            if (s, seed) in done:
                continue
            ts = time.time()
            _, vh, th = run_fold(T, tr, va, te, y, args.rung, hp, seed, device, epochs)
            yte = y[te]
            from sklearn.metrics import roc_auc_score
            ep_h = int(np.nanargmax(vh))
            aucs = [roc_auc_score(yte, th[e]) if len(np.unique(yte)) > 1 else np.nan
                    for e in range(th.shape[0])]
            ep_p = int(np.nanargmax(aucs))
            row = {"rung": args.rung, "site": s, "seed": seed, "n_test": len(te),
                   "hp": json.dumps(hp), "epoch_honest": ep_h, "epoch_paper1": ep_p,
                   "secs": round(time.time() - ts, 1)}
            for tag, ep in (("honest", ep_h), ("paper1", ep_p)):
                for k2, v2 in compute_metrics(yte, th[ep]).items():
                    row[f"{tag}_{k2}"] = round(v2, 4) if isinstance(v2, float) else v2
            row["inflation"] = round(row["paper1_roc_auc"] - row["honest_roc_auc"], 4)
            pd.DataFrame([row]).to_csv(res, mode="a",
                                       header=not os.path.exists(res), index=False)
            print(f"  seed {seed:4d} honest={row['honest_roc_auc']:.3f} "
                  f"paper1={row['paper1_roc_auc']:.3f} (infl {row['inflation']:+.3f}) "
                  f"acc={row['honest_accuracy']:.3f} f1={row['honest_f1']:.3f} "
                  f"({row['secs']}s)")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\ntotal {(time.time()-t0)/60:.1f} min -> {res}")
    if os.path.exists(res):
        report(pd.read_csv(res))
        print("\nanchors: edge-SVM 0b=0.658 | fusion 0c=0.664 | robust4 flat=0.630 "
              "| sMRI 0a=0.593 | motion floor=0.561")


if __name__ == "__main__":
    main()
