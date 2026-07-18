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
from baselines import COHORT, make_estimator, score_of, func_block  # noqa: E402
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
    print(f"physio {physio.shape}  naive {naive.shape}")

    print("=== naive2 [mean_fc,std_fc] ===")
    dn = eval_block("naive2", naive.astype(np.float64), coh, y, sites, tsites)
    print("=== physio4 [fALFF,Hurst,wCC,PC] ===")
    dp = eval_block("physio4", physio.astype(np.float64), coh, y, sites, tsites)

    out = pd.concat([dn, dp], ignore_index=True)
    out.to_csv(os.path.join(OUTDIR, "node_compare.csv"), index=False)

    print("\n=== node-feature comparison (mean over sites) ===")
    for nm, d in [("naive2", dn), ("physio4", dp)]:
        print(f"  {nm:8} ROC {d.roc_auc.mean():.4f} +/- {d.roc_auc.std():.4f}   "
              f"PR {d.pr_auc.mean():.4f}")
    print("  ref 0b edge (6105 Fisher-z SVM)   ROC 0.6580   (from stage4_ladder.csv)")

    a = dp.set_index("site").roc_auc; b = dn.set_index("site").roc_auc
    common = a.index.intersection(b.index)
    if len(common) >= 3:
        w, p = wilcoxon(a.loc[common], b.loc[common])
        print(f"\nphysio4 vs naive2: median dAUC={np.median(a.loc[common]-b.loc[common]):+.3f}  "
              f"Wilcoxon W={w:.1f} p={p:.4f}  wins {int((a.loc[common]>b.loc[common]).sum())}/{len(common)}")
        print("MDE band 0.06-0.07 -> "
              + ("physio features change the result" if abs(np.median(a.loc[common]-b.loc[common])) >= 0.06
                 else "difference below MDE (no material change)"))
    print(f"\nwrote {os.path.join(OUTDIR,'node_compare.csv')}")


if __name__ == "__main__":
    main()
