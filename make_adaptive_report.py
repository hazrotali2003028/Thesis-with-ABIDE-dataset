"""
Adaptive (per-subject) HeteroGNN vs the group-level HeteroGNN.

Regenerates results/adaptive_vs_nonadaptive_report.txt from the two nested-LOSO
result files. The adaptive model rebuilds its edges per subject (kNN, k searched)
and searches depth {2,3,4}; the question is whether that fixes the group-level
adjacency null. It does not: Delta AUC ~ +0.003, Wilcoxon p ~ 0.94.

Timing (the secs column, per fold-seed fit) is reported here too, since it was
the user's ask and is what a reviewer needs to judge reproducibility cost.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
ADAPT = os.path.join(OUT, "nested_loso_adaptive_results.csv")
BASE = os.path.join(OUT, "nested_loso_results.csv")
REPORT = os.path.join(OUT, "adaptive_vs_nonadaptive_report.txt")


def fmt_hms(s):
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def main():
    a = pd.read_csv(ADAPT)
    a["secs"] = pd.to_numeric(a["secs"], errors="coerce")
    a["layers"] = a["hp"].apply(lambda s: json.loads(s)["layers"])
    a["k"] = a["hp"].apply(lambda s: json.loads(s)["k"])

    # nested-LOSO search shape, mirroring train_nested_loso_adaptive.py:
    #   HP_GRID = {hidden:[64], layers:[2,3,4], k:[5,10], dropout:[0.3]} -> 6 combos
    #   inner GroupKFold folds default = 3
    n_hp_combos = 6
    inner_folds = 3

    o = pd.read_csv(BASE)
    oc = "test_auc" if "test_auc" in o.columns else \
        [c for c in o.columns if "auc" in c.lower()][0]

    A = a.groupby("site")["test_auc"].mean()
    O = o.groupby("site")[oc].mean()
    sites = sorted(A.index)

    per = a.groupby("site").agg(
        adaptive=("test_auc", "mean"),
        sd=("test_auc", "std"),
        layers=("layers", lambda x: x.mode().iat[0]),
        k=("k", lambda x: x.mode().iat[0]),
        sec_seed=("secs", "mean"),
        sec_site=("secs", "sum"),
    )
    per["nonadaptive"] = O
    per["delta"] = per["adaptive"] - per["nonadaptive"]

    w, p = wilcoxon(A.loc[sites], O.loc[sites])
    wins = int((per["delta"] > 0).sum())
    total_secs = a["secs"].sum()

    lines = []
    lines.append("Adaptive (per-subject) HeteroGNN vs group-level HeteroGNN")
    lines.append("=" * 72)
    lines.append(f"sites paired: {len(sites)}   seeds/site: {a['seed'].nunique()}   "
                 f"fits: {len(a)}")
    lines.append(f"adaptive     LOSO AUC: {A.mean():.4f} +/- {A.std():.4f}")
    lines.append(f"non-adaptive LOSO AUC: {O.mean():.4f} +/- {O.std():.4f}")
    lines.append(f"mean delta (adaptive - group): {per['delta'].mean():+.4f}   "
                 f"adaptive wins {wins}/{len(sites)} sites")
    lines.append(f"Wilcoxon signed-rank (paired by site): W={w:.1f}  p={p:.4f}")
    lines.append("")
    lines.append("VERDICT: rebuilding the graph per-subject does NOT beat the")
    lines.append("group-level adjacency (p=0.94). The subject-invariant edge tensor")
    lines.append("was a real defect but not the binding constraint; the ceiling is the")
    lines.append("232 structural features (~0.56), still below svm_rbf (0.604). Report")
    lines.append("as a negative control that closes the 'you never tried a real graph'")
    lines.append("objection, not as a graph that adds value.")
    lines.append("")
    lines.append("Inner-CV hyperparameter selection (over 100 fits):")
    lc = a["layers"].value_counts().sort_index().to_dict()
    kc = a["k"].value_counts().sort_index().to_dict()
    lines.append(f"  layers chosen: {lc}")
    lines.append(f"  k (neighbours) chosen: {kc}")
    lines.append("")

    # per-site table (accuracy + timing), sorted by adaptive AUC desc
    tbl = per.sort_values("adaptive", ascending=False)
    lines.append("Per-site AUC and runtime (5 seeds/site):")
    hdr = (f"{'site':<9}{'adaptive':>9}{'sd':>7}{'group':>7}{'delta':>8}"
           f"{'lyr':>5}{'k':>4}{'sec/seed':>10}{'site_total':>12}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s, r in tbl.iterrows():
        lines.append(
            f"{s:<9}{r.adaptive:>9.3f}{r.sd:>7.3f}{r.nonadaptive:>7.3f}"
            f"{r.delta:>+8.3f}{int(r.layers):>5}{int(r.k):>4}"
            f"{r.sec_seed:>10.1f}{fmt_hms(r.sec_site):>12}")
    lines.append("-" * len(hdr))
    lines.append("")
    # The `secs` column times ONLY the outer refit+eval (timer starts inside the
    # seed loop). It excludes the inner HP search and the per-fold build_fold
    # (ComBat + graph). Reported wall-clock was ~21h; the timed slice is ~6.4h.
    inner_fits = len(sites) * n_hp_combos * inner_folds
    outer_fits = len(a)
    lines.append("Runtime summary:")
    lines.append(f"  TIMED (secs column) = outer refit+eval only : "
                 f"{fmt_hms(total_secs)}  ({total_secs:.0f}s over {outer_fits} fits)")
    lines.append(f"  per-fit  mean/min/max                       : "
                 f"{a['secs'].mean():.1f}s / {a['secs'].min():.1f}s / "
                 f"{a['secs'].max():.1f}s")
    slow = per["sec_site"].idxmax()
    fast = per["sec_site"].idxmin()
    lines.append(f"  slowest / fastest site (timed)              : "
                 f"{slow} {fmt_hms(per.loc[slow,'sec_site'])} / "
                 f"{fast} {fmt_hms(per.loc[fast,'sec_site'])}")
    lines.append("")
    lines.append("  NOT timed by the secs column, but part of wall-clock:")
    lines.append(f"    inner HP search : {inner_fits} fits "
                 f"({len(sites)} sites x {n_hp_combos} configs x {inner_folds} "
                 f"GroupKFold folds) -- {inner_fits/outer_fits:.1f}x the outer count")
    lines.append("    per-fold build_fold (ComBat fit + per-subject graph build)")
    lines.append(f"  => OBSERVED WALL-CLOCK ~= 21h (single process). The {fmt_hms(total_secs)}"
                 " above is the")
    lines.append("     final-fit slice only, not total runtime.")
    lines.append("")
    lines.append("  note: timed per-fit cost is inverse to fold size -- a larger held-out")
    lines.append("  site leaves a smaller LOSO training set, hence fewer batches/epoch.")
    lines.append("")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
