"""
Stage 3 (plan v2.1 -- section 3.2, F1-F4): build the functional connectivity cache.

Reads dualgraph/cohort_final.csv, and for each subject:
  F2  truncate to the first T* volumes                     (T* = 116, equal length)
  --  drop the site-systematic dead ROIs (FOV clipping), globally, so a clipped
      cerebellum cannot become learnable site structure    (116 -> R_keep nodes)
  F3  Ledoit-Wolf shrinkage correlation                    (T=116, 116 ROIs is
      rank-<=115 singular; LW also makes isolated dead ROIs non-NaN)
  F4  Fisher-z, zero the diagonal

Writes cache/fc_z.npy [Nf, R_keep, R_keep] float32 + cache/fc_manifest.json
(kept/dropped ROI indices, subject order, T*, shape, SHA256).

Does NOT use nilearn: nilearn's ConnectivityMeasure(cov_estimator=LedoitWolf) is a
thin wrapper over the same sklearn estimator, and nilearn is not installed. The
4-line fc_ledoitwolf() below is the identical math.

Run:  python dualgraph/build_fc.py                 # drop ROIs dead in >=10 subjects
      python dualgraph/build_fc.py --keep-all-roi  # ablation: keep all 116
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROIS_DIR = os.path.join(ROOT, "abide", "abide_fmri", "Outputs", "cpac",
                        "nofilt_noglobal", "rois_aal")
COHORT = os.path.join(HERE, "cohort_final.csv")
CACHE = os.path.join(HERE, "cache")


def fc_ledoitwolf(ts):
    """Shrinkage correlation matrix from a [T, R] timeseries."""
    cov = LedoitWolf(store_precision=False).fit(ts).covariance_
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1.0
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def load_ts(fid, tstar):
    a = np.loadtxt(os.path.join(ROIS_DIR, f"{fid}_rois_aal.1D"))
    return a[:tstar]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tstar", type=int, default=116)
    ap.add_argument("--dead-min", type=int, default=10,
                    help="drop ROIs dead (zero-variance) in >= this many subjects")
    ap.add_argument("--keep-all-roi", action="store_true")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    coh = pd.read_csv(COHORT)
    fids = coh.FILE_ID.tolist()
    print(f"cohort: {len(fids)} subjects  T*={args.tstar}")

    # ---- pass 1: which ROIs are site-systematically dead (on the truncated ts) ----
    dead_counter = Counter()
    for fid in fids:
        ts = load_ts(fid, args.tstar)
        for r in np.flatnonzero(ts.std(axis=0) == 0):
            dead_counter[int(r)] += 1
    if args.keep_all_roi:
        drop = []
    else:
        drop = sorted(r for r, n in dead_counter.items() if n >= args.dead_min)
    keep = [r for r in range(116) if r not in drop]
    print(f"dead-ROI counts (top): {dead_counter.most_common(8)}")
    print(f"dropping {len(drop)} ROIs (dead in >={args.dead_min} subj): {drop}")
    print(f"kept ROIs: {len(keep)}")

    # ---- pass 2: FC per subject on kept ROIs ----
    R = len(keep)
    fc = np.zeros((len(fids), R, R), dtype=np.float32)
    n_nan = 0
    resid_dead = Counter()
    for i, fid in enumerate(fids):
        ts = load_ts(fid, args.tstar)[:, keep]
        for r in np.flatnonzero(ts.std(axis=0) == 0):
            resid_dead[int(keep[r])] += 1
        corr = fc_ledoitwolf(ts)
        z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
        np.fill_diagonal(z, 0.0)
        if np.isnan(z).any():
            n_nan += 1
        fc[i] = z.astype(np.float32)

    out = os.path.join(CACHE, "fc_z.npy")
    np.save(out, fc)
    sha = hashlib.sha256(fc.tobytes()).hexdigest()
    n_edges = R * (R - 1) // 2

    manifest = {
        "shape": list(fc.shape),
        "tstar": args.tstar,
        "n_subjects": len(fids),
        "kept_roi": keep,
        "dropped_roi": drop,
        "dead_min": args.dead_min,
        "n_upper_tri_edges": n_edges,
        "subject_order_FILE_ID": fids,
        "sha256": sha,
        "fc_range": [float(fc.min()), float(fc.max())],
    }
    with open(os.path.join(CACHE, "fc_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {out}  shape={fc.shape}  {fc.nbytes/1e6:.1f} MB")
    print(f"upper-tri edges: {n_edges}")
    print(f"any NaN subjects: {n_nan}  (LW should give 0)")
    print(f"fc_z range: [{fc.min():.3f}, {fc.max():.3f}]  mean|z|={np.abs(fc).mean():.3f}")
    print(f"sha256: {sha[:16]}...")

    # ---- Stage 3 gate: residual dead ROIs must NOT be site-systematic ----
    print("\n=== gate: residual dead ROIs (after drop) ===")
    if not resid_dead:
        print("  none -- gate PASS")
    else:
        rd = pd.DataFrame([(k, v) for k, v in resid_dead.items()],
                          columns=["roi", "n_subj"]).sort_values("n_subj",
                                                                 ascending=False)
        print(rd.to_string(index=False))
        print(f"  max residual = {rd.n_subj.max()} subjects "
              f"({'PASS' if rd.n_subj.max() < args.dead_min else 'REVIEW'} "
              f"-- LW keeps these non-NaN, isolated)")


if __name__ == "__main__":
    main()
