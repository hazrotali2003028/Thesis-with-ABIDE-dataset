"""
Batch-extract functional node features for the frozen 794-subject cohort.

Writes to dualgraph/node_features/:
  node_feat_physio.npy  [794, 111, 4]  [fALFF, Hurst, wCC, PC]   (this upgrade)
  node_feat_naive.npy   [794, 111, 2]  [mean_fc, std_fc]         (the baseline)
  nodefeat_manifest.json

Reuses cache/fc_z.npy for the topological features (wCC, PC) and the naive
summaries; loads each .1D only for the temporal features (fALFF, Hurst), truncated
to T*=116 and reduced to the same 111 kept ROIs as Stage 3.

TR is site-specific (ABIDE-I acquisition protocol table) -- fALFF needs it.

Run:  python dualgraph/extract_nodefeat.py
"""

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from node_features import extract_node_features, DEAD_ROI_0BASED, N_NODES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROIS_DIR = os.path.join(ROOT, "abide", "abide_fmri", "Outputs", "cpac",
                        "nofilt_noglobal", "rois_aal")
COHORT = os.path.join(HERE, "cohort_final.csv")
FC = os.path.join(HERE, "cache", "fc_z.npy")
OUTDIR = os.path.join(HERE, "node_features")
TSTAR = 116
KEEP = [i for i in range(116) if i not in DEAD_ROI_0BASED]

# ABIDE-I site repetition times (seconds), from the acquisition protocol table.
SITE_TR = {
    "CALTECH": 2.0, "CMU": 2.0, "KKI": 2.5, "LEUVEN_1": 1.6667, "LEUVEN_2": 1.6667,
    "MAX_MUN": 3.0, "NYU": 2.0, "OHSU": 2.5, "OLIN": 1.5, "PITT": 1.5, "SBL": 2.2,
    "SDSU": 2.0, "STANFORD": 2.0, "TRINITY": 2.0, "UCLA_1": 3.0, "UCLA_2": 3.0,
    "UM_1": 2.0, "UM_2": 2.0, "USM": 2.0, "YALE": 2.0,
}


def naive_node_feats(fc_z: np.ndarray) -> np.ndarray:
    """[R,R] -> [R,2] [mean_fc, std_fc] over each node's off-diagonal edges."""
    m = ~np.eye(fc_z.shape[0], dtype=bool)
    rows = [fc_z[i, m[i]] for i in range(fc_z.shape[0])]
    mean = np.array([r.mean() for r in rows])
    std = np.array([r.std() for r in rows])
    return np.column_stack([mean, std]).astype(np.float32)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    coh = pd.read_csv(COHORT)
    fc = np.load(FC)
    assert len(coh) == len(fc)

    physio = np.zeros((len(coh), N_NODES, 4), dtype=np.float32)
    naive = np.zeros((len(coh), N_NODES, 2), dtype=np.float32)
    import time
    t0 = time.time()
    for i, row in coh.iterrows():
        tr = SITE_TR[row.SITE_ID]
        ts = np.loadtxt(os.path.join(ROIS_DIR, f"{row.FILE_ID}_rois_aal.1D"))
        ts = ts[:TSTAR][:, KEEP]
        physio[i] = extract_node_features(ts, fc[i], tr)
        naive[i] = naive_node_feats(fc[i])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(coh)}  ({(time.time()-t0)/60:.1f} min)")

    np.save(os.path.join(OUTDIR, "node_feat_physio.npy"), physio)
    np.save(os.path.join(OUTDIR, "node_feat_naive.npy"), naive)
    man = {
        "physio_shape": list(physio.shape),
        "physio_cols": ["fALFF", "Hurst", "wCC", "PC"],
        "naive_shape": list(naive.shape),
        "naive_cols": ["mean_fc", "std_fc"],
        "tstar": TSTAR, "n_nodes": N_NODES,
        "subject_order_FILE_ID": coh.FILE_ID.tolist(),
        "physio_sha256": hashlib.sha256(physio.tobytes()).hexdigest(),
        "physio_col_means": physio.reshape(-1, 4).mean(0).round(4).tolist(),
        "physio_col_finite": bool(np.isfinite(physio).all()),
    }
    with open(os.path.join(OUTDIR, "nodefeat_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"\nphysio {physio.shape} finite={np.isfinite(physio).all()} "
          f"means={man['physio_col_means']}")
    print(f"naive  {naive.shape}")
    print(f"wrote -> {OUTDIR}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
