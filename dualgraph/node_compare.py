"""
Node-feature comparison: physiological-4 vs naive-2, Stage-4 protocol.

Answers "does higher-level node featurisation change the result?" by running the
exact Stage 4 nested-LOSO SVM on flattened per-node features:

  naive2   [mean_fc, std_fc]            -> 111 x 2 = 222 features
  physio4  [fALFF, Hurst, wCC, PC]      -> 111 x 4 = 444 features

Same protocol as baselines.py: 17-site nested LOSO, inner 3-fold
StratifiedGroupKFold(SITE_ID), per-fold ComBat (protect [DX,age,sex,FD]) fit on
train rows only, ROC + PR per site. Reference point: the 0b EDGE model (SVM on the
6105 Fisher-z) = 0.658, read from stage4_ladder.csv (not re-run).

Run:  python dualgraph/node_compare.py            # full
      python dualgraph/node_compare.py --sites 3  # smoke
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from baselines import COHORT, FC, make_estimator, score_of, func_block  # noqa: E402
from motion_gates import loso_oof                                    # noqa: E402

NF = os.path.join(HERE, "node_features")
OUTDIR = os.path.join(HERE, "results")


def eval_block(name, X, coh, y, sites, tsites):
    """LOSO with per-fold ComBat on flattened node features; per-site ROC+PR."""
    fold = func_block(X, coh, use_combat=True)     # generic: ComBats any [N,F]
    oof, site_auc = loso_oof("linsvm", fold, y, sites, [0.001, 0.01, 0.1, 1], tsites)
    rows = []
    for s in (tsites if tsites else sorted(np.unique(sites))):
        ms = sites == s
        rows.append({"block": name, "site": s, "n_test": int(ms.sum()),
                     "roc_auc": round(site_auc[s], 4),
                     "pr_auc": round(average_precision_score(y[ms], oof[ms]), 4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    coh = pd.read_csv(COHORT)
    y = (coh.DX_GROUP == 1).to_numpy(int)
    sites = coh.SITE_ID.to_numpy()
    tsites = sorted(np.unique(sites))[:args.sites] if args.sites else None

    physio = np.load(os.path.join(NF, "node_feat_physio.npy")).reshape(len(coh), -1)
    naive = np.load(os.path.join(NF, "node_feat_naive.npy")).reshape(len(coh), -1)

    # robust4 = [strength, eigen-centrality, within-module-z, PC], FC-only (fast)
    rob_path = os.path.join(NF, "node_feat_robust.npy")
    if os.path.exists(rob_path):
        robust = np.load(rob_path).reshape(len(coh), -1)
    else:
        from node_features import extract_robust_features
        fc = np.load(FC)
        rob = np.stack([extract_robust_features(fc[i]) for i in range(len(coh))])
        np.save(rob_path, rob.astype(np.float32))
        robust = rob.reshape(len(coh), -1)
    print(f"naive {naive.shape}  physio {physio.shape}  robust {robust.shape}")

    print("=== naive2 [mean_fc,std_fc] ===")
    dn = eval_block("naive2", naive.astype(np.float64), coh, y, sites, tsites)
    print("=== physio4 [fALFF,Hurst,wCC,PC] ===")
    dp = eval_block("physio4", physio.astype(np.float64), coh, y, sites, tsites)
    print("=== robust4 [strength,eigcen,wmz,PC] ===")
    dr = eval_block("robust4", robust.astype(np.float64), coh, y, sites, tsites)

    out = pd.concat([dn, dp, dr], ignore_index=True)
    out.to_csv(os.path.join(OUTDIR, "node_compare.csv"), index=False)

    print("\n=== three-way node-feature comparison (mean over sites) ===")
    for nm, d in [("naive2", dn), ("physio4", dp), ("robust4", dr)]:
        print(f"  {nm:8} ROC {d.roc_auc.mean():.4f} +/- {d.roc_auc.std():.4f}   "
              f"PR {d.pr_auc.mean():.4f}")
    print("  ref 0b edge (6105 Fisher-z SVM)   ROC 0.6580   (from stage4_ladder.csv)")

    def pair(A, B, nA, nB):
        a = A.set_index("site").roc_auc; b = B.set_index("site").roc_auc
        c = a.index.intersection(b.index)
        w, p = wilcoxon(a.loc[c], b.loc[c])
        d = np.median(a.loc[c] - b.loc[c])
        tag = "MATERIAL" if abs(d) >= 0.06 else "below MDE"
        print(f"  {nA:8} vs {nB:8}: median dAUC={d:+.3f}  p={p:.4f}  "
              f"wins {int((a.loc[c]>b.loc[c]).sum())}/{len(c)}  [{tag}]")

    print("\npaired Wilcoxon (MDE band 0.06-0.07):")
    pair(dr, dn, "robust4", "naive2")
    pair(dr, dp, "robust4", "physio4")
    pair(dp, dn, "physio4", "naive2")
    print(f"\nwrote {os.path.join(OUTDIR,'node_compare.csv')}")


if __name__ == "__main__":
    main()
