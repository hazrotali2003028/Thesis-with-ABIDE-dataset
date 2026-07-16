"""
What does Paper 1's ComBat-on-everything actually cost? (plan Stage 4 leakage)

Paper 1 harmonizes df_all -- train + val + test concatenated -- with DX_GROUP
"preserved" as a covariate, BEFORE any fold is cut. ComBat preserves a covariate
by subtracting its effect and adding it back:

    Y_adj = Z_adj * sqrt(var_pooled) + stand_mean
    stand_mean = grand_mean + X_cov @ B_cov          <- X_cov contains DX

So every ASD subject receives +B_dx[j] on feature j. When X_cov includes the DX
of TEST subjects, the label is written into the test features as a rank-1 shift
that any linear model can read off. This is a label leak, not merely a
distributional one.

This measures it under the HONEST epoch-selection rule, so the ONLY thing that
varies is the harmonization:

    train_only  ComBat fit on training sites; test site forward-applied,
                its DX never used            (this project's method)
    leaky_dx    ComBat fit on ALL sites with DX of every subject in the design
                                             (Paper 1's df_all method)
    leaky_nodx  ComBat fit on ALL sites, but DX excluded from the design
                                             (isolates site leakage from label leakage)

The third condition separates the two effects: leaky_nodx - train_only is the
cost of letting the test site influence the harmonization; leaky_dx - leaky_nodx
is the cost of letting test LABELS in.

Outputs (results/):
    combat_leak.csv
    combat_leak_report.txt

Run:
    python check_combat_leak.py
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

from hetero_data import feature_matrix, harmonize_and_scale
from combat import combat_fit, combat_apply_train

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def leaky_split(df, train_idx, test_idx, include_dx):
    """Paper 1's method: ComBat over ALL sites at once, then cut the fold."""
    Y, _ = feature_matrix(df)
    dx = df["label"].to_numpy(float)
    age = df["AGE_AT_SCAN"].to_numpy(float)
    sex = df["SEX"].to_numpy(float)
    site = df["SITE_ID"].to_numpy()

    if include_dx:
        X = np.column_stack([dx, age, sex])
        nolabel = [1, 2]
    else:
        X = np.column_stack([age, sex])
        nolabel = [0, 1]

    est = combat_fit(Y, site, X, nolabel)          # <- test site included in the fit
    H = combat_apply_train(Y, site, X, est)

    Htr, Hte = H[train_idx], H[test_idx]
    mu, sd = Htr.mean(0), Htr.std(0)
    sd[sd == 0] = 1.0
    return ((Htr - mu) / sd, (Hte - mu) / sd,
            df.loc[train_idx, "label"].to_numpy(),
            df.loc[test_idx, "label"].to_numpy())


def models(seed=42):
    m = {"logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
         "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=seed)}
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
    print(f"subjects {len(df)} | sites {len(sites)}\n", flush=True)

    rows = []
    t0 = time.time()
    for i, site in enumerate(sites, 1):
        tm = (df["SITE_ID"] == site).to_numpy()
        trm = ~tm

        conds = {}
        Ztr, Zte, ytr, yte, _ = harmonize_and_scale(df, trm, tm, use_combat=True)
        conds["train_only"] = (Ztr, Zte, ytr, yte)
        conds["leaky_nodx"] = leaky_split(df, trm, tm, include_dx=False)
        conds["leaky_dx"] = leaky_split(df, trm, tm, include_dx=True)

        for cond, (a, b, ya, yb) in conds.items():
            for name, clf in models().items():
                clf.fit(a, ya)
                p = clf.predict_proba(b)[:, 1]
                auc = roc_auc_score(yb, p) if len(np.unique(yb)) > 1 else np.nan
                rows.append({"condition": cond, "model": name, "site": site,
                             "test_auc": auc, "n_test": int(tm.sum())})
        print(f"  [{i:2d}/{len(sites)}] {site:10s} ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "combat_leak.csv"), index=False)

    piv = res.pivot_table(index=["model", "site"], columns="condition",
                          values="test_auc").reset_index()

    L = ["What Paper 1's ComBat-on-everything is worth",
         "=" * 72,
         "Honest epoch selection throughout. Only the harmonization varies.",
         "",
         f"{'model':10s} {'train_only':>11} {'leaky_nodx':>11} {'leaky_dx':>10} "
         f"{'site leak':>10} {'LABEL leak':>11}"]
    for m in sorted(piv["model"].unique()):
        s = piv[piv["model"] == m]
        t, ln, ld = s["train_only"].mean(), s["leaky_nodx"].mean(), s["leaky_dx"].mean()
        L.append(f"{m:10s} {t:11.4f} {ln:11.4f} {ld:10.4f} "
                 f"{ln-t:+10.4f} {ld-ln:+11.4f}")

    L += ["", "Significance vs train_only (Wilcoxon, paired over 20 sites):"]
    for m in sorted(piv["model"].unique()):
        s = piv[piv["model"] == m]
        for c in ("leaky_nodx", "leaky_dx"):
            W, p = wilcoxon(s[c], s["train_only"])
            L.append(f"  {m:10s} {c:11s} delta = {(s[c]-s['train_only']).mean():+.4f}  "
                     f"p = {p:.4g}")

    all_site = np.mean([piv[piv.model == m]["leaky_nodx"].mean()
                        - piv[piv.model == m]["train_only"].mean()
                        for m in piv["model"].unique()])
    all_lab = np.mean([piv[piv.model == m]["leaky_dx"].mean()
                       - piv[piv.model == m]["leaky_nodx"].mean()
                       for m in piv["model"].unique()])
    L += ["",
          f"Mean cost of letting the test site into the fit : {all_site:+.4f}",
          f"Mean cost of letting test LABELS into the fit   : {all_lab:+.4f}",
          "",
          "Per-site AUC:",
          piv.to_string(index=False, float_format=lambda v: f"{v:.3f}")]

    rep = "\n".join(L)
    with open(os.path.join(OUT_DIR, "combat_leak_report.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep)


if __name__ == "__main__":
    main()
