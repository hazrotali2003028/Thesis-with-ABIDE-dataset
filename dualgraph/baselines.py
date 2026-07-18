"""
Stage 4 (plan v2.1 section 7) -- the baseline ladder, rungs -2 .. 0c.

The gate the whole project hinges on: does functional connectivity beat the head-
motion floor?  Rung -2 (motion) is the floor for every fMRI claim; if 0b - (-2) <
MDE, fMRI adds nothing over fidgeting and no architecture rescues it.

Rungs (all on the SAME frozen 794 / 17-site dual cohort, so site-AUCs are paired):
  -2   LogReg on [func_mean_fd]                 head motion alone      (~0.58 expected)
  -1   LogReg on [age, sex, FIQ, FD]            phenotype, no site dummies
  0a   SVC-rbf  on 232 structural (ComBat)      sMRI linear
  0b   LinearSVM on 6105 Fisher-z (edge-ComBat) fMRI linear
  0c   LinearSVM on [232 || 6105]               linear fusion

Protocol (section 6): nested LOSO-site. Outer = 17 folds (one site out). Inner =
3-fold StratifiedGroupKFold(groups=SITE_ID) for HP selection -> refit on all
training sites -> predict the held-out site once. ComBat / edge-ComBat / scaler
are fit on the fold's TRAINING rows only, refit inside every inner split too.

Deterministic estimators (SVC/LinearSVC/LogReg), so no seed loop is needed -- the
plan's 5 seeds are for the stochastic GNNs. Unit = per-site AUC -> n = 17 paired.
Every rung reports ROC-AUC AND PR-AUC (section 8; ASD prevalence = 357/794 = 0.45).

Run:  python dualgraph/baselines.py                # full ladder
      python dualgraph/baselines.py --sites 3      # smoke: first 3 held-out sites
      python dualgraph/baselines.py --no-combat    # scaler-only ablation (A4 off)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from hetero_data import harmonize_and_scale, feature_matrix          # noqa: E402
from combat import combat_fit, combat_apply_train, combat_apply_unseen  # noqa: E402

COHORT = os.path.join(HERE, "cohort_final.csv")
FC = os.path.join(HERE, "cache", "fc_z.npy")
FEATURES = os.path.join(ROOT, "features", "abide_features_raw.csv")
OUTDIR = os.path.join(HERE, "results")


# ---------- per-fold preprocessing closures (fit on the train mask only) ----------
def scaler_block(X):
    def fold(tr, te):
        sc = StandardScaler().fit(X[tr])
        return sc.transform(X[tr]), sc.transform(X[te])
    return fold


def struct_block(struct_df, use_combat):
    def fold(tr, te):
        Ztr, Zte, *_ = harmonize_and_scale(struct_df, tr, te, use_combat=use_combat)
        return Ztr, Zte
    return fold


def func_block(edges, coh, use_combat):
    dx = (coh.DX_GROUP == 1).to_numpy(float)
    age = coh.AGE_AT_SCAN.to_numpy(float)
    sex = coh.SEX.to_numpy(float)
    fd = coh.func_mean_fd.to_numpy(float)
    site = coh.SITE_ID.to_numpy()
    Xp = np.column_stack([dx, age, sex, fd])
    NOLABEL = [1, 2, 3]                       # age, sex, FD available on unseen; dx not

    def fold(tr, te):
        Ytr, Yte = edges[tr], edges[te]
        if use_combat:
            est = combat_fit(Ytr, site[tr], Xp[tr], NOLABEL)
            Htr = combat_apply_train(Ytr, site[tr], Xp[tr], est)
            Hte = combat_apply_unseen(Yte, Xp[te], est)
        else:
            Htr, Hte = Ytr.copy(), Yte.copy()
        mu, sd = Htr.mean(0), Htr.std(0)
        sd[sd == 0] = 1.0
        return (Htr - mu) / sd, (Hte - mu) / sd
    return fold


def fusion_block(struct_fold, func_fold):
    def fold(tr, te):
        s_tr, s_te = struct_fold(tr, te)
        f_tr, f_te = func_fold(tr, te)
        return np.hstack([s_tr, f_tr]), np.hstack([s_te, f_te])
    return fold


# ---------- estimators + scores ----------
def make_estimator(kind, hp):
    if kind == "logreg":
        return LogisticRegression(C=hp, max_iter=2000)
    if kind == "svc_rbf":
        return SVC(kernel="rbf", C=hp[0], gamma=hp[1])
    if kind == "linsvm":
        return LinearSVC(C=hp, max_iter=5000, dual="auto")
    raise ValueError(kind)


def score_of(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


# ---------- nested-LOSO evaluation of one rung ----------
def eval_rung(name, kind, fold_fn, y, sites, grid, test_sites=None):
    usites = test_sites if test_sites is not None else sorted(np.unique(sites))
    rows = []
    for s in usites:
        te = sites == s
        tr = ~te
        # inner HP selection: StratifiedGroupKFold on training sites only
        idx_tr = np.flatnonzero(tr)
        sgkf = StratifiedGroupKFold(n_splits=3)
        agg = {repr(hp): [] for hp in grid}
        for itr, iva in sgkf.split(idx_tr, y[tr], groups=sites[tr]):
            m_itr = np.zeros(len(y), bool); m_itr[idx_tr[itr]] = True
            m_iva = np.zeros(len(y), bool); m_iva[idx_tr[iva]] = True
            Xtr, Xva = fold_fn(m_itr, m_iva)
            for hp in grid:
                clf = make_estimator(kind, hp).fit(Xtr, y[m_itr])
                a = roc_auc_score(y[m_iva], score_of(clf, Xva)) \
                    if len(np.unique(y[m_iva])) > 1 else np.nan
                agg[repr(hp)].append(a)
        best = max(grid, key=lambda hp: np.nanmean(agg[repr(hp)]))
        # refit on all training sites, predict held-out site once
        Xtr, Xte = fold_fn(tr, te)
        clf = make_estimator(kind, best).fit(Xtr, y[tr])
        sc = score_of(clf, Xte)
        roc = roc_auc_score(y[te], sc)
        pr = average_precision_score(y[te], sc)
        rows.append({"rung": name, "site": s, "n_test": int(te.sum()),
                     "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
                     "best_hp": json.dumps(best)})
        print(f"  {name:4} {s:9} roc={roc:.3f} pr={pr:.3f} hp={best}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=0, help="smoke: only first N held-out sites")
    ap.add_argument("--no-combat", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    use_combat = not args.no_combat

    coh = pd.read_csv(COHORT)
    fc = np.load(FC)
    assert len(coh) == len(fc), "cohort / fc_z length mismatch"
    R = fc.shape[1]
    iu = np.triu_indices(R, 1)
    edges = fc[:, iu[0], iu[1]].astype(np.float64)          # [Nf, 6105]
    y = (coh.DX_GROUP == 1).to_numpy(int)                   # ASD = 1
    sites = coh.SITE_ID.to_numpy()
    print(f"cohort {len(coh)}  ASD {y.sum()} / TD {(y==0).sum()}  "
          f"prevalence {y.mean():.3f}  edges {edges.shape[1]}  combat={use_combat}")

    # structural 232-vector aligned to cohort order
    feat = pd.read_csv(FEATURES)
    sdf = coh[["SUB_ID"]].merge(feat, on="SUB_ID", how="left").reset_index(drop=True)
    sdf["label"] = y                                        # keep ComBat protect consistent
    assert not feature_matrix(sdf)[0].shape[0] != len(coh)
    struct_fold = struct_block(sdf, use_combat)
    func_fold = func_block(edges, coh, use_combat)

    # FD / phenotype raw blocks (impute FIQ with global median; scaler is per fold)
    fiq = coh.FIQ.fillna(coh.FIQ.median()).to_numpy(float)
    X_fd = coh[["func_mean_fd"]].to_numpy(float)
    X_ph = np.column_stack([coh.AGE_AT_SCAN, coh.SEX, fiq, coh.func_mean_fd]).astype(float)

    # smoke: hold out only the first N sites, but TRAIN on all 17
    test_sites = sorted(np.unique(sites))[:args.sites] if args.sites else None

    rungs = [
        ("-2", "logreg", scaler_block(X_fd), [0.01, 0.1, 1, 10]),
        ("-1", "logreg", scaler_block(X_ph), [0.01, 0.1, 1, 10]),
        ("0a", "svc_rbf", struct_fold, [(1, "scale"), (10, "scale"), (1, 0.01)]),
        ("0b", "linsvm", func_fold, [0.001, 0.01, 0.1, 1]),
        ("0c", "linsvm", fusion_block(struct_fold, func_fold), [0.001, 0.01, 0.1]),
    ]

    all_rows = []
    for name, kind, fold_fn, grid in rungs:
        print(f"\n=== rung {name} ({kind}) ===")
        all_rows += eval_rung(name, kind, fold_fn, y, sites, grid, test_sites)

    df = pd.DataFrame(all_rows)
    out = os.path.join(OUTDIR, "stage4_ladder.csv")
    df.to_csv(out, index=False)

    # ---- summary + the Stage-4 gate ----
    print("\n=== Stage 4 ladder (mean over sites) ===")
    piv = df.groupby("rung").agg(roc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"),
                                 pr=("pr_auc", "mean"), n=("site", "size"))
    order = ["-2", "-1", "0a", "0b", "0c"]
    piv = piv.reindex([r for r in order if r in piv.index])
    print(piv.round(4).to_string())

    if {"-2", "0b"} <= set(df.rung.unique()):
        a = df[df.rung == "0b"].set_index("site")["roc_auc"]
        b = df[df.rung == "-2"].set_index("site")["roc_auc"]
        common = a.index.intersection(b.index)
        w, p = wilcoxon(a.loc[common], b.loc[common])
        d = float((a.loc[common] - b.loc[common]).median())
        print(f"\nGATE 0b vs -2 (fMRI vs motion): median dAUC={d:+.3f}  "
              f"Wilcoxon W={w:.1f} p={p:.4f}  (n={len(common)} sites)")
        print("MDE planning band 0.06-0.07 -> "
              + ("CLEARS" if d >= 0.06 else "BELOW MDE: fMRI adds nothing over motion (kill criterion)"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
