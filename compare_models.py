"""
HeteroGNN vs baselines: paired significance testing (plan Stage 9 / RQ2).

The comparison is PAIRED BY SITE. Each model produced an AUC on the same 20
held-out sites through the same folds and the same train-only ComBat, so the
20 differences are matched pairs and a Wilcoxon signed-rank test is the right
instrument. A higher mean AUC on its own is not evidence: Paper 1's GAT beat
the MLP on the mean and still came out at p = 0.984.

Reported per comparison:
    delta        mean AUC difference (GNN - baseline)
    W, p         Wilcoxon signed-rank on the 20 paired site AUCs
    p_holm       Holm-Bonferroni across the family of baselines
    rank_biserial  effect size in [-1, 1]; sign follows delta
    n_sites_won  how many of the 20 sites the GNN actually won

Outputs (results/):
    model_comparison.csv
    model_comparison_report.txt

Run:
    python compare_models.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
GNN_CSV = os.path.join(OUT_DIR, "nested_loso_results.csv")
BASE_CSV = os.path.join(OUT_DIR, "baseline_per_site_auc.csv")


def rank_biserial(diff):
    """Effect size for the Wilcoxon signed-rank test: (W+ - W-) / (W+ + W-)."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def main():
    if not os.path.exists(GNN_CSV):
        raise SystemExit(f"missing {GNN_CSV} -- run train_nested_loso.py first")
    if not os.path.exists(BASE_CSV):
        raise SystemExit(f"missing {BASE_CSV} -- run train_baselines.py first")

    gnn = pd.read_csv(GNN_CSV)
    gnn_site = gnn.groupby("site")["test_auc"].mean().rename("hetero_gnn")

    base = pd.read_csv(BASE_CSV)
    base_wide = base.pivot(index="site", columns="model", values="test_auc")

    joined = base_wide.join(gnn_site, how="inner").dropna(subset=["hetero_gnn"])
    n_sites = len(joined)
    print(f"paired on {n_sites} sites\n")
    if n_sites < len(base_wide):
        print(f"NOTE: GNN has results for {n_sites} of {len(base_wide)} sites; "
              "comparison uses the intersection.\n")

    rows = []
    models = [c for c in joined.columns if c != "hetero_gnn"]
    for m in models:
        d = (joined["hetero_gnn"] - joined[m]).to_numpy()
        if np.allclose(d, 0):
            W, p = np.nan, 1.0
        else:
            W, p = wilcoxon(joined["hetero_gnn"], joined[m])
        rows.append({
            "baseline": m,
            "baseline_auc": joined[m].mean(),
            "gnn_auc": joined["hetero_gnn"].mean(),
            "delta": d.mean(),
            "W": W,
            "p": p,
            "rank_biserial": rank_biserial(d),
            "n_sites_won": int((d > 0).sum()),
            "n_sites": n_sites,
        })

    res = pd.DataFrame(rows).sort_values("baseline_auc", ascending=False)
    res["p_holm"] = holm(res["p"].to_numpy())
    res = res[["baseline", "baseline_auc", "gnn_auc", "delta", "W", "p",
               "p_holm", "rank_biserial", "n_sites_won", "n_sites"]]

    out_csv = os.path.join(OUT_DIR, "model_comparison.csv")
    res.to_csv(out_csv, index=False)

    strongest = res.iloc[0]
    beat_all = bool((res["delta"] > 0).all() and (res["p_holm"] < 0.05).all())

    with open(os.path.join(OUT_DIR, "model_comparison_report.txt"), "w") as f:
        f.write("HeteroGNN vs baselines - Wilcoxon signed-rank, paired by site\n")
        f.write("=" * 72 + "\n")
        f.write(f"sites paired: {n_sites}\n")
        f.write(f"HeteroGNN LOSO AUC: {joined['hetero_gnn'].mean():.4f} "
                f"+/- {joined['hetero_gnn'].std():.4f}\n\n")
        f.write(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        f.write("\n\n")
        f.write(f"Strongest baseline: {strongest['baseline']} "
                f"(AUC {strongest['baseline_auc']:.4f})\n")
        f.write(f"  delta = {strongest['delta']:+.4f}, p = {strongest['p']:.4f}, "
                f"p_holm = {strongest['p_holm']:.4f}\n\n")
        if beat_all:
            f.write("VERDICT: the HeteroGNN beats every baseline with Holm-adjusted\n"
                    "p < 0.05. RQ2 is supported on this evidence.\n")
        else:
            f.write("VERDICT: the HeteroGNN does NOT significantly beat every baseline.\n"
                    "RQ2 is not supported. Report this outcome honestly: a graph that\n"
                    "ties a tabular model is a real result, and Paper 1 already found\n"
                    "GAT ~= MLP at p = 0.984. Do not claim the graph adds value.\n")
        f.write("\nPer-site AUC (paired):\n")
        f.write(joined.to_string(float_format=lambda v: f"{v:.3f}"))
        f.write("\n")

    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out_csv}")
    print(f"wrote {os.path.join(OUT_DIR, 'model_comparison_report.txt')}")
    print("\nVERDICT:", "GNN beats all baselines (Holm p<0.05)" if beat_all
          else "GNN does NOT significantly beat all baselines")


if __name__ == "__main__":
    main()
