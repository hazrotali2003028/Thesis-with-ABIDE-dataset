"""
Node coordinates for the heterogeneous graph (plan Stage 6/7, "MNI encoding").

What these coordinates actually are — read before citing them
------------------------------------------------------------
The plan asks for MNI xyz per node. True MNI would require either fsaverage or
each subject's talairach.xfm, and neither is in the archives. Rather than invent
atlas numbers, this derives centroids from the cohort itself:

  - cortical:    mean vertex position of each Desikan parcel on ?h.pial
  - subcortical: voxel centroid of each aseg label

Both are expressed in FreeSurfer surface RAS (tkrRAS), which is a head-centred
frame defined by the 256^3 conformed volume, then averaged over subjects. Pial
coordinates are natively tkrRAS; aseg voxel coordinates are mapped with the
volume's own vox2ras_tkr, so BOTH node types land in one common frame. Scanner
RAS is deliberately NOT used: it depends on head position in the bore and is not
comparable across subjects.

So these are cohort-mean tkrRAS centroids, NOT MNI coordinates. That is fine for
their actual job here — a fixed positional encoding, identical for every subject,
whose only requirement is a consistent anatomical frame. Do not report them as
MNI in the paper, and do not reuse them for anything needing true stereotaxic
space (the spin test needs spherical surface coordinates, not these).

Output:
    features/node_coords.csv   96 rows: node, node_type, x, y, z, n_subjects

Run:
    python build_coords.py [n_subjects]
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.freesurfer.io import read_annot, read_label, read_geometry

from extract_features import (
    ROOT, MRI_DIR, LGI_DIR, LABEL_DIR, OUT_DIR,
    ASEG_LABELS, FEATURE_REGIONS, DESIKAN_34,
)

N_DEFAULT = 150
NAME_TO_LABEL = {v: k for k, v in ASEG_LABELS.items()}


def subject_coords(sid):
    """Centroid of every node for one subject, in tkrRAS."""
    try:
        out = {}

        # ── subcortical: aseg voxel centroids -> tkrRAS ──
        img = nib.load(os.path.join(MRI_DIR, sid, "mri", "aseg.mgz"))
        data = np.asanyarray(img.dataobj)
        vox2tkr = img.header.get_vox2ras_tkr()
        for name in FEATURE_REGIONS:
            lab = NAME_TO_LABEL[name]
            vox = np.argwhere(data == lab)
            if len(vox) == 0:
                out[name] = np.full(3, np.nan)
                continue
            c = vox.mean(axis=0)
            out[name] = (vox2tkr @ np.array([c[0], c[1], c[2], 1.0]))[:3]

        # ── cortical: mean pial vertex position per parcel (already tkrRAS) ──
        for hemi in ("lh", "rh"):
            coords, _ = read_geometry(os.path.join(LGI_DIR, sid, "surf", f"{hemi}.pial"))
            labels, _, names = read_annot(
                os.path.join(LABEL_DIR, sid, "label", f"{hemi}.aparc.annot"))
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            cortex_idx = read_label(
                os.path.join(LABEL_DIR, sid, "label", f"{hemi}.cortex.label"))
            cmask = np.zeros(len(coords), dtype=bool)
            cmask[cortex_idx] = True

            name_to_idx = {n: i for i, n in enumerate(names)}
            for parcel in DESIKAN_34:
                key = f"{hemi}_{parcel}"
                idx = name_to_idx.get(parcel)
                if idx is None:
                    out[key] = np.full(3, np.nan)
                    continue
                m = (labels == idx) & cmask
                out[key] = coords[m].mean(axis=0) if m.any() else np.full(3, np.nan)

        return sid, out, None
    except Exception as exc:
        return sid, None, f"{type(exc).__name__}: {exc}"


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT

    feat = pd.read_csv(os.path.join(OUT_DIR, "abide_features_raw.csv"))
    ok = feat[feat["qc_pass"]]["folder_id"].tolist()
    # spread across sites so no single site dominates the mean centroid
    rng = np.random.default_rng(0)
    sids = sorted(rng.choice(ok, size=min(n_want, len(ok)), replace=False).tolist())
    print(f"Averaging node centroids over {len(sids)} qc_pass subjects\n")

    acc, fails = {}, []
    with Pool(5) as pool:
        for i, (sid, out, err) in enumerate(pool.imap_unordered(subject_coords, sids), 1):
            if err:
                fails.append((sid, err))
                print(f"  FAILED {sid}: {err}")
                continue
            for k, v in out.items():
                acc.setdefault(k, []).append(v)
            if i % 25 == 0:
                print(f"  {i}/{len(sids)}", flush=True)

    rows = []
    for name in FEATURE_REGIONS:
        arr = np.vstack(acc[name])
        rows.append({"node": name, "node_type": "subcortical",
                     **dict(zip("xyz", np.nanmean(arr, axis=0))),
                     "n_subjects": int((~np.isnan(arr[:, 0])).sum())})
    for hemi in ("lh", "rh"):
        for parcel in DESIKAN_34:
            key = f"{hemi}_{parcel}"
            arr = np.vstack(acc[key])
            rows.append({"node": key, "node_type": "cortical",
                         **dict(zip("xyz", np.nanmean(arr, axis=0))),
                         "n_subjects": int((~np.isnan(arr[:, 0])).sum())})

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "node_coords.csv")
    df.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(df)} nodes)")

    # ── sanity: anatomy must come out the right way round ──
    print("\nSanity checks (tkrRAS: +x=right, +y=anterior, +z=superior)")
    lh = df[df.node.str.startswith("lh_")]
    rh = df[df.node.str.startswith("rh_")]
    print(f"  lh cortical mean x = {lh.x.mean():+7.2f}  (expect negative)")
    print(f"  rh cortical mean x = {rh.x.mean():+7.2f}  (expect positive)")
    lsub = df[df.node.str.startswith("L_")]
    rsub = df[df.node.str.startswith("R_")]
    print(f"  L_ subcortical x   = {lsub.x.mean():+7.2f}  (expect negative)")
    print(f"  R_ subcortical x   = {rsub.x.mean():+7.2f}  (expect positive)")

    def g(n, a):
        return float(df.loc[df.node == n, a].iloc[0])
    print(f"  frontalpole y      = {g('lh_frontalpole','y'):+7.2f} vs "
          f"occipital(lateraloccipital) y = {g('lh_lateraloccipital','y'):+7.2f}  (frontal must be anterior)")
    print(f"  paracentral z      = {g('lh_paracentral','z'):+7.2f} vs "
          f"temporalpole z = {g('lh_temporalpole','z'):+7.2f}  (paracentral must be superior)")
    print(f"  Brainstem z        = {g('Brainstem','z'):+7.2f}  (expect inferior/negative)")
    lr = np.abs(lh.x.mean() + rh.x.mean())
    print(f"  |lh.x + rh.x|      = {lr:7.2f}  (near 0 => left/right symmetric)")


if __name__ == "__main__":
    main()
