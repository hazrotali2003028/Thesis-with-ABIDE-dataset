"""
Is ComBat removing the ASD signal? (Paper 1 diagnostics, "Suspect 4")

Everything downstream rests on the harmonized features. If ComBat is stripping
diagnosis variance along with site variance, then the ~0.60 ceiling is an
artefact of MY pipeline, not biology -- and every conclusion about the graph is
drawn on damaged data.

The test is a direct A/B under the identical LOSO protocol:
    harmonized : train-only ComBat with DX protected, then train-only scaler
    raw        : train-only scaler ONLY, no ComBat

If raw >= harmonized, ComBat is hurting. If harmonized >= raw, ComBat is doing
its job and the ceiling is real.

Paired by site, Wilcoxon signed-rank, as with the model comparison.

Outputs (results/):
    harmonization_check.csv        per model x site x condition
    harmonization_check_report.txt verdict

Run:
    python check_harmonization.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from hetero_data import harmonize_and_scale, feature_matrix

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def models(seed=42):
    m = {
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", C=1.0, probability=True,
                       class_weight="balanced", random_state=seed),
    }
    if HAS_XGB:
        m["xgboost"] = XGBClassifier(n_estimators=400, max_depth=4,
                                     learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss",
                                     tree_method="hist", random_state=seed, n_jobs=-1)
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    sites = sorted(df["SITE_ID"].unique())
    print(f"subjects {len(df)} | sites {len(sites)} | features 232\n")

    rows = []
    t0 = time.time()
    for i, site in enumerate(sites, 1):
        test_mask = (df["SITE_ID"] == site).to_numpy()
        train_mask = ~test_mask
        for cond, use_combat in (("harmonized", True), ("raw", False)):
            Ztr, Zte, ytr, yte, _ = harmonize_and_scale(
                df, train_mask, test_mask, use_combat=use_combat)
            for name, clf in models().items():
                clf.fit(Ztr, ytr)
                p = clf.predict_proba(Zte)[:, 1]
                auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
                rows.append({"model": name, "condition": cond,
                             "site": site, "test_auc": auc, "n_test": len(yte)})
        print(f"  [{i:2d}/{len(sites)}] {site:10s} ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "harmonization_check.csv"), index=False)

    piv = res.pivot_table(index=["model", "site"], columns="condition",
                          values="test_auc").reset_index()

    lines = []
    lines.append("Does ComBat remove the ASD signal?  (A/B under identical LOSO)")
    lines.append("=" * 68)
    lines.append(f"subjects {len(df)} | sites {len(sites)} | features 232\n")
    lines.append(f"{'model':10s} {'harmonized':>11} {'raw':>8} {'delta':>8} "
                 f"{'p':>8} {'raw wins':>9}")

    verdicts = []
    for m in sorted(piv["model"].unique()):
        s = piv[piv["model"] == m]
        h, r = s["harmonized"].to_numpy(), s["raw"].to_numpy()
        d = h - r
        W, p = wilcoxon(h, r) if not np.allclose(d, 0) else (np.nan, 1.0)
        lines.append(f"{m:10s} {h.mean():11.4f} {r.mean():8.4f} {d.mean():+8.4f} "
                     f"{p:8.4f} {int((r > h).sum()):6d}/{len(s)}")
        verdicts.append(d.mean())

    lines.append("")
    mean_delta = float(np.mean(verdicts))
    if mean_delta > 0.01:
        lines.append("VERDICT: harmonization HELPS. ComBat is not the bottleneck;")
        lines.append("the ~0.60 ceiling reflects the data, not the pipeline.")
    elif mean_delta < -0.01:
        lines.append("VERDICT: raw beats harmonized. ComBat is REMOVING ASD signal")
        lines.append("(Paper 1 'Suspect 4' confirmed). Every downstream conclusion")
        lines.append("is drawn on damaged features and must be re-run.")
    else:
        lines.append("VERDICT: harmonization is roughly NEUTRAL for accuracy.")
        lines.append("It is not destroying signal, and it is not buying accuracy")
        lines.append("either -- its value would have to be lower cross-site variance.")

    # Does ComBat at least deliver what it promises: less cross-site spread?
    lines.append("\nCross-site spread of per-site AUC (ComBat's actual purpose):")
    for m in sorted(piv["model"].unique()):
        s = piv[piv["model"] == m]
        lines.append(f"  {m:10s} sd(harmonized)={s['harmonized'].std():.4f}  "
                     f"sd(raw)={s['raw'].std():.4f}")

    lines.append("\nPer-site AUC:")
    lines.append(piv.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "harmonization_check_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nwrote {os.path.join(OUT_DIR, 'harmonization_check.csv')}")
    print(f"wrote {os.path.join(OUT_DIR, 'harmonization_check_report.txt')}")


if __name__ == "__main__":
    main()
