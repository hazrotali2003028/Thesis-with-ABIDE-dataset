"""
Baseline models under the SAME LOSO protocol as the HeteroGNN (plan Stage 9).

Why this file must exist before any graph claim
-----------------------------------------------
Paper 1 found GAT ~= MLP at p = 0.984. A HeteroGNN AUC means nothing on its own;
it only means something relative to a strong tabular floor on identical folds.
So every model here consumes the exact same output of
hetero_data.harmonize_and_scale() -- same 979 subjects, same 232 features, same
train-only ComBat, same train-only scaler, same 20 LOSO folds. The ONLY thing
that varies is the estimator.

Models (plan Stage 9 "mandatory" list, plus the no-graph neural control):
    dummy       majority class            -- absolute floor
    logreg      L2 logistic regression    -- mandatory linear baseline
    svm_linear  linear SVM
    svm_rbf     RBF SVM                   -- non-linear classical
    rf          random forest             -- ensemble
    xgboost     gradient boosting         -- mandatory strong tabular baseline
    mlp         2-layer MLP, no graph     -- isolates the graph's contribution

Deterministic models run once; stochastic ones (rf, xgboost, mlp) run 5 seeds,
matching the GNN protocol so the Wilcoxon comparison is paired like-for-like.

Outputs (all written to results/):
    baseline_results.csv    one row per (model, site, seed) with full metrics
    baseline_summary.csv    per-model mean/std over the 20 site-level AUCs
    baseline_report.txt     human-readable ranking + the floor to beat

Run:
    python train_baselines.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from hetero_data import harmonize_and_scale

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")

SEEDS = [42, 123, 456, 789, 1234]

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def make_models(seed, class_weight):
    """Fresh estimators. class_weight is computed per fold, matching the GNN's
    per-fold weighted cross-entropy (site ASD rates span ~39-65%)."""
    m = {
        "dummy": (DummyClassifier(strategy="prior"), False),
        "logreg": (LogisticRegression(max_iter=5000, C=1.0,
                                      class_weight="balanced"), False),
        "svm_linear": (SVC(kernel="linear", C=1.0, probability=True,
                           class_weight="balanced", random_state=seed), False),
        "svm_rbf": (SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=seed), False),
        "rf": (RandomForestClassifier(n_estimators=500, max_depth=None,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=seed), True),
        "mlp": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                              early_stopping=True, random_state=seed), True),
    }
    if HAS_XGB:
        m["xgboost"] = (XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=class_weight, eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1), True)
    return m


def metrics(y, p):
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    if len(np.unique(y)) > 1:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
    else:
        sens = spec = np.nan
    return auc, acc, sens, spec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not HAS_XGB:
        print("WARNING: xgboost not installed -- the plan's mandatory strong "
              "tabular baseline will be MISSING from the comparison.\n")

    df = pd.read_csv(FEATURES)
    df = df[df["qc_pass"]].reset_index(drop=True)
    sites = sorted(df["SITE_ID"].unique())
    print(f"subjects: {len(df)}  sites: {len(sites)}  features: 232")
    print(f"models: {', '.join(make_models(0, 1.0).keys())}\n")

    rows = []
    t0 = time.time()
    for si, site in enumerate(sites, 1):
        test_mask = (df["SITE_ID"] == site).to_numpy()
        train_mask = ~test_mask

        # Identical preprocessing to the GNN: fitted on training sites only,
        # forward-applied to the held-out site.
        Ztr, Zte, ytr, yte, _ = harmonize_and_scale(df, train_mask, test_mask)
        cw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

        for name in make_models(SEEDS[0], cw):
            seeds = SEEDS if make_models(SEEDS[0], cw)[name][1] else [SEEDS[0]]
            for seed in seeds:
                clf, _ = make_models(seed, cw)[name]
                ts = time.time()
                clf.fit(Ztr, ytr)
                p = clf.predict_proba(Zte)[:, 1]
                auc, acc, sens, spec = metrics(yte, p)
                rows.append({
                    "model": name, "site": site, "seed": seed,
                    "test_auc": auc, "accuracy": acc,
                    "sensitivity": sens, "specificity": spec,
                    "n_test": len(yte), "n_train": len(ytr),
                    "secs": round(time.time() - ts, 2),
                })
        print(f"  [{si:2d}/{len(sites)}] {site:10s} n_test={len(yte):3d}  "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    res = pd.DataFrame(rows)
    res_path = os.path.join(OUT_DIR, "baseline_results.csv")
    res.to_csv(res_path, index=False)

    # Per-site mean over seeds, then mean/std over sites -- the LOSO estimate.
    per_site = res.groupby(["model", "site"])["test_auc"].mean().reset_index()
    summ = (per_site.groupby("model")["test_auc"]
            .agg(mean_auc="mean", std_auc="std", n_sites="count")
            .reset_index().sort_values("mean_auc", ascending=False))
    for m in summ["model"]:
        sub = res[res["model"] == m]
        summ.loc[summ["model"] == m, "accuracy"] = sub["accuracy"].mean()
        summ.loc[summ["model"] == m, "sensitivity"] = sub["sensitivity"].mean()
        summ.loc[summ["model"] == m, "specificity"] = sub["specificity"].mean()
    summ_path = os.path.join(OUT_DIR, "baseline_summary.csv")
    summ.to_csv(summ_path, index=False)

    per_site.to_csv(os.path.join(OUT_DIR, "baseline_per_site_auc.csv"), index=False)

    best = summ.iloc[0]
    with open(os.path.join(OUT_DIR, "baseline_report.txt"), "w") as f:
        f.write("ABIDE I - Baseline LOSO results (plan Stage 9)\n")
        f.write("=" * 64 + "\n")
        f.write(f"subjects {len(df)} | sites {len(sites)} | features 232\n")
        f.write("Preprocessing identical to the HeteroGNN: train-only ComBat + scaler.\n")
        f.write("Per-site AUC averaged over seeds, then averaged over sites.\n\n")
        f.write(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        f.write("\n\n")
        f.write(f"FLOOR TO BEAT: {best['model']} at AUC "
                f"{best['mean_auc']:.4f} +/- {best['std_auc']:.4f}\n")
        f.write("The HeteroGNN must beat this by a Wilcoxon signed-rank test over\n"
                "the 20 paired site-level AUCs, not merely on the mean.\n")
        f.write("\nPer-site AUC by model:\n")
        f.write(per_site.pivot(index="site", columns="model", values="test_auc")
                .to_string(float_format=lambda v: f"{v:.3f}"))
        f.write("\n")

    print(f"\ntotal {(time.time()-t0)/60:.1f} min")
    print(f"\nwrote {res_path}")
    print(f"wrote {summ_path}")
    print(f"wrote {os.path.join(OUT_DIR, 'baseline_per_site_auc.csv')}")
    print(f"wrote {os.path.join(OUT_DIR, 'baseline_report.txt')}\n")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nFLOOR TO BEAT: {best['model']} AUC {best['mean_auc']:.4f} "
          f"+/- {best['std_auc']:.4f}")


if __name__ == "__main__":
    main()
