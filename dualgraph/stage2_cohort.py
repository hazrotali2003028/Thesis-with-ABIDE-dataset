"""
Stage 2 (plan v2.1 -- section 2): freeze the dual sMRI-cap-fMRI cohort.

Computes every number the plan left as an empty slot (Nf, DX split, T*), writes
cohort_final.csv, and reports the two feasibility issues found while doing it:

  1. T* rule: the plan text says "15-20th percentile" (= 146 here), but that drops
     2 sites and 112 subjects, and contradicts the plan's own hardcoded 116 (F1
     rejects T<116) and its 17-outer-fold count. T*=116 drops ONLY OHSU (all at
     T=78), keeps 17 sites / 794 subjects (97%). Default here is 116; justify it
     as "largest T retaining every site but the single shortest-scan one."
  2. Dead ROIs are site-systematic (cerebellar AAL indices ~86/100/101/106/107) =
     FOV clipping. Left in a top-k/threshold graph they become site structure ->
     a site-classification leak. Reported per ROI so Stage 3 can drop them.

Ledoit-Wolf (Stage 3 F3) regularizes the dead ROIs to non-NaN, so NO subject is
lost to them; the concern is fake edges, not missing data.

Run (from anywhere):  python dualgraph/stage2_cohort.py
                       python dualgraph/stage2_cohort.py --tstar 146
"""

import argparse
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROIS = os.path.join(ROOT, "abide", "abide_fmri", "Outputs", "cpac",
                    "nofilt_noglobal", "rois_aal", "*.1D")
PHENO = os.path.join(ROOT, "abide", "Phenotypic_V1_0b_preprocessed1.csv")
SMRI = os.path.join(ROOT, "features", "abide_features_raw.csv")
OUT = os.path.join(HERE, "cohort_final.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tstar", type=int, default=116)
    ap.add_argument("--min-site-n", type=int, default=20)
    args = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(ROIS)):
        fid = os.path.basename(f).replace("_rois_aal.1D", "")
        a = np.loadtxt(f)
        assert a.ndim == 2 and a.shape[1] == 116, f"malformed {fid}: {a.shape}"
        dead = int((a.std(axis=0) == 0).sum())
        recs.append({"FILE_ID": fid, "T": a.shape[0], "dead_roi": dead})
    tf = pd.DataFrame(recs)
    print(f"F1: loaded {len(tf)} timeseries, all [T,116], "
          f"{(tf.dead_roi>0).sum()} with >=1 dead ROI")

    ph = pd.read_csv(PHENO)
    cols = ["FILE_ID", "SUB_ID", "SITE_ID", "DX_GROUP",
            "AGE_AT_SCAN", "SEX", "FIQ", "func_mean_fd"]
    c = tf.merge(ph[cols], on="FILE_ID", how="inner")
    sm = pd.read_csv(SMRI)
    sm_ids = set(sm[sm.qc_pass == True].SUB_ID.astype(int))
    c = c[c.SUB_ID.astype(int).isin(sm_ids)].copy()
    print(f"dual sMRI-cap-fMRI: {len(c)} subjects")

    vc = c.SITE_ID.value_counts()
    dropped_small = sorted(vc[vc <= args.min_site_n].index)
    c = c[c.SITE_ID.isin(vc[vc > args.min_site_n].index)].copy()
    print(f"after n>{args.min_site_n}: N={len(c)} sites={c.SITE_ID.nunique()} "
          f"(dropped {dropped_small})")
    print(f"  DX split: ASD={int((c.DX_GROUP==1).sum())} "
          f"TD={int((c.DX_GROUP==2).sum())}")
    print(f"  T percentiles [10,15,20,25,50]: "
          f"{np.round(np.percentile(c['T'], [10,15,20,25,50]),1)}")

    cf = c[c["T"] >= args.tstar].copy()
    dropped_T = c[c["T"] < args.tstar]
    print(f"\n=== FROZEN COHORT (T* = {args.tstar}) ===")
    print(f"Nf = {len(cf)}   sites = {cf.SITE_ID.nunique()}   "
          f"split = {int((cf.DX_GROUP==1).sum())} ASD / "
          f"{int((cf.DX_GROUP==2).sum())} TD")
    print(f"dropped-by-T = {len(dropped_T)} "
          f"(sites fully lost: {sorted(set(dropped_T.SITE_ID)-set(cf.SITE_ID))})")

    keep = ["SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN",
            "SEX", "FIQ", "func_mean_fd", "T", "dead_roi"]
    cf[keep].to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(cf)} rows)")


if __name__ == "__main__":
    main()
