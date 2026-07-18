"""
Stage 9 motion gates -- runnable subset (G1, G3). The question: does the fMRI
model classify ASD, or is it a head-motion detector?

FD is confounded (r(FD,DX)=+0.138, ASD move more). Clearing Stage 4 (0b>motion
floor) is necessary but NOT sufficient, because edge-ComBat PROTECTED func_mean_fd,
i.e. it deliberately kept motion variance in the features. These gates attack that.

G1 (plan 9.2) -- score vs motion. Collect out-of-fold P(ASD) from 0b for every
subject (each predicted once, on its held-out site), then:
  * r(score, FD)              -- if high, the output is a motion signal
  * r(score, label)           -- should be > r(score, FD) if it tracks diagnosis
  * partial r(score, label | FD) -- does it STILL predict ASD after removing FD?
  * reference: r(motion-model score, FD), which is ~1 by construction
  * per-site r(score, FD): how many sites show a significant motion coupling

G3 (plan 9.4) -- FD removed. Re-run 0b after linearly regressing FD out of every
edge (fit on the train fold, applied to test) and dropping FD from the ComBat
protected set. If AUC survives, the discriminative signal is not linear motion.

Deferred (need the GNN / AAL centroids): G2 motion-matched subsample, G4 Power
short/long-range signature + attention alignment, G5 deployment metrics.

Run:  python dualgraph/motion_gates.py            # full (re-runs 0b twice + motion)
      python dualgraph/motion_gates.py --sites 3  # smoke
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr, wilcoxon, t as tdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from combat import combat_fit, combat_apply_train, combat_apply_unseen  # noqa: E402
from baselines import COHORT, FC, make_estimator, score_of              # noqa: E402

OUTDIR = os.path.join(HERE, "results")


def func_fold(edges, coh, protect_fd=True, fd_regress=False):
    """Per-fold FC preprocessing. protect_fd keeps FD variance in ComBat;
    fd_regress linearly removes FD from every edge (train-fit) first."""
    dx = (coh.DX_GROUP == 1).to_numpy(float)
    age = coh.AGE_AT_SCAN.to_numpy(float)
    sex = coh.SEX.to_numpy(float)
    fd = coh.func_mean_fd.to_numpy(float)
    site = coh.SITE_ID.to_numpy()
    if protect_fd:
        Xp = np.column_stack([dx, age, sex, fd]); nolabel = [1, 2, 3]
    else:
        Xp = np.column_stack([dx, age, sex]); nolabel = [1, 2]

    def fold(tr, te):
        Ytr, Yte = edges[tr].copy(), edges[te].copy()
        if fd_regress:                       # remove linear FD per edge, train-fit
            f_tr = fd[tr]; f_te = fd[te]
            A = np.column_stack([np.ones_like(f_tr), f_tr])   # [n,2]
            beta, *_ = np.linalg.lstsq(A, Ytr, rcond=None)    # [2, E]
            Ytr = Ytr - A @ beta
            Yte = Yte - np.column_stack([np.ones_like(f_te), f_te]) @ beta
        est = combat_fit(Ytr, site[tr], Xp[tr], nolabel)
        Htr = combat_apply_train(Ytr, site[tr], Xp[tr], est)
        Hte = combat_apply_unseen(Yte, Xp[te], est)
        mu, sd = Htr.mean(0), Htr.std(0); sd[sd == 0] = 1.0
        return (Htr - mu) / sd, (Hte - mu) / sd
    return fold


def loso_oof(kind, fold_fn, y, sites, grid, test_sites=None):
    """Nested-LOSO; return per-subject out-of-fold score and per-site AUC."""
    oof = np.full(len(y), np.nan)
    usites = test_sites if test_sites is not None else sorted(np.unique(sites))
    site_auc = {}
    for s in usites:
        te = sites == s; tr = ~te
        idx_tr = np.flatnonzero(tr)
        sgkf = StratifiedGroupKFold(n_splits=3)
        agg = {repr(hp): [] for hp in grid}
        for itr, iva in sgkf.split(idx_tr, y[tr], groups=sites[tr]):
            m_itr = np.zeros(len(y), bool); m_itr[idx_tr[itr]] = True
            m_iva = np.zeros(len(y), bool); m_iva[idx_tr[iva]] = True
            Xtr, Xva = fold_fn(m_itr, m_iva)
            for hp in grid:
                clf = make_estimator(kind, hp).fit(Xtr, y[m_itr])
                agg[repr(hp)].append(roc_auc_score(y[m_iva], score_of(clf, Xva)))
        best = max(grid, key=lambda hp: np.nanmean(agg[repr(hp)]))
        Xtr, Xte = fold_fn(tr, te)
        clf = make_estimator(kind, best).fit(Xtr, y[tr])
        sc = score_of(clf, Xte)
        oof[te] = sc
        site_auc[s] = roc_auc_score(y[te], sc)
        print(f"  {kind:8} {s:9} auc={site_auc[s]:.3f} hp={best}")
    return oof, site_auc


def partial_r(a, b, c):
    """partial correlation r(a,b | c)."""
    rab = pearsonr(a, b)[0]; rac = pearsonr(a, c)[0]; rbc = pearsonr(b, c)[0]
    r = (rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2))
    n = len(a); df = n - 3
    tval = r * np.sqrt(df / (1 - r**2))
    p = 2 * tdist.sf(abs(tval), df)
    return r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    coh = pd.read_csv(COHORT)
    fc = np.load(FC)
    R = fc.shape[1]; iu = np.triu_indices(R, 1)
    edges = fc[:, iu[0], iu[1]].astype(np.float64)
    y = (coh.DX_GROUP == 1).to_numpy(int)
    fd = coh.func_mean_fd.to_numpy(float)
    sites = coh.SITE_ID.to_numpy()
    tsites = sorted(np.unique(sites))[:args.sites] if args.sites else None

    print("=== rung -2 (motion only) OOF ===")
    from baselines import scaler_block
    oof_m, auc_m = loso_oof("logreg", scaler_block(fd[:, None]), y, sites,
                            [0.01, 0.1, 1, 10], tsites)
    print("=== 0b (fMRI, FD protected) OOF ===")
    oof_b, auc_b = loso_oof("linsvm", func_fold(edges, coh, protect_fd=True),
                            y, sites, [0.001, 0.01, 0.1, 1], tsites)
    print("=== 0b_fdreg (FD regressed out) OOF ===")
    oof_r, auc_r = loso_oof("linsvm",
                            func_fold(edges, coh, protect_fd=False, fd_regress=True),
                            y, sites, [0.001, 0.01, 0.1, 1], tsites)

    m = ~np.isnan(oof_b)
    print("\n================ G1: score vs motion vs label ================")
    print(f"pooled N = {m.sum()}")
    print(f"  r(0b score, FD)          = {pearsonr(oof_b[m], fd[m])[0]:+.3f} "
          f"(p={pearsonr(oof_b[m], fd[m])[1]:.3g})   [Spearman "
          f"{spearmanr(oof_b[m], fd[m])[0]:+.3f}]")
    print(f"  r(0b score, label)       = {pointbiserialr(y[m], oof_b[m])[0]:+.3f} "
          f"(p={pointbiserialr(y[m], oof_b[m])[1]:.3g})")
    pr, pp = partial_r(y[m].astype(float), oof_b[m], fd[m])
    print(f"  partial r(label, 0b | FD)= {pr:+.3f} (p={pp:.3g})   <-- signal after removing FD")
    print(f"  reference r(motion score,FD)= {pearsonr(oof_m[m], fd[m])[0]:+.3f} "
          f"(the pure-motion model, ~ceiling)")

    print("\n  per-site r(0b score, FD):")
    rows = []
    for s in (tsites if tsites else sorted(np.unique(sites))):
        ms = (sites == s)
        rr, pv = pearsonr(oof_b[ms], fd[ms])
        rows.append({"site": s, "n": int(ms.sum()), "auc_0b": round(auc_b[s], 3),
                     "auc_fdreg": round(auc_r[s], 3),
                     "r_score_fd": round(rr, 3), "p_score_fd": round(pv, 4)})
    g1 = pd.DataFrame(rows)
    sig = (g1.p_score_fd < 0.05).sum()
    print(g1.to_string(index=False))
    print(f"  sites with significant score~FD coupling: {sig}/{len(g1)}")

    print("\n================ G3: does AUC survive FD removal? ================")
    a = np.array([auc_b[s] for s in auc_b]); b = np.array([auc_r[s] for s in auc_r])
    w, p = wilcoxon(a, b)
    print(f"  0b (FD kept)   mean AUC = {a.mean():.3f}")
    print(f"  0b_fdreg       mean AUC = {b.mean():.3f}")
    print(f"  paired dAUC (kept-reg) median = {np.median(a-b):+.3f}  Wilcoxon p={p:.4f}")
    verdict = "signal SURVIVES FD removal -> not just motion" if b.mean() >= a.mean() - 0.03 \
        else "AUC drops materially -> motion contributes"
    print(f"  VERDICT: {verdict}")

    g1.to_csv(os.path.join(OUTDIR, "motion_gates.csv"), index=False)
    print(f"\nwrote {os.path.join(OUTDIR,'motion_gates.csv')}")


if __name__ == "__main__":
    main()
