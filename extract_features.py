"""
ABIDE I — Paper 3: HeteroGNN Feature Extraction
================================================
Implements Stage 2 (multi-channel extraction) and the measurement half of
Stage 3 (quality control) of ABIDE_HeteroGNN_Research_Plan_v2.

Reads precomputed FreeSurfer 6 outputs with nibabel only (no FreeSurfer
software required).

Input archives (one row per subject, joined on the trailing SUB_ID integer):
    abide_mri_brain/{sid}/mri/aseg.mgz              -> 28 subcortical volumes
    abide_native_desc/{sid}/surf/?h.thickness       -> cortical thickness
    abide_native_desc/{sid}/surf/?h.area            -> white-surface area
    abide_native_desc/{sid}/surf/?h.curv            -> quality index only
    abide_freesurfer6_lgi/{sid}/surf/?h.pial_lgi    -> gyrification
    abide_freesurfer6_lgi/{sid}/surf/?h.pial        -> Euler gate only
    abide_label/{sid}/label/?h.aparc.annot          -> Desikan-68 region map
    abide_label/{sid}/label/?h.cortex.label         -> medial-wall mask
    Phenotypic_V1_0b.csv                            -> DX_GROUP, age, sex, site

Output (features/):
    abide_features_raw.csv     one row per subject, 232 features + QC + phenotype
    feature_names.txt          column groups, in graph-node order
    qc_report.txt              per-gate exclusion counts
    extraction_log.csv         per-subject success/failure with reasons

Design notes:
  - Subjects are streamed one at a time in worker processes. Nothing larger
    than a single subject is ever held in memory (peak ~150 MB/worker), which
    is what keeps this inside a 7.7 GB box.
  - QC gates are measured here but NOT applied. Every subject is written out
    with boolean flags so the modelling stage decides exclusions. Gates that
    need cohort statistics (|z|>5, vertex-count deviation) are label-independent,
    so computing them cohort-wide does not leak diagnosis.

Run:
    python extract_features.py
"""

import os
import sys
import csv
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.freesurfer.io import read_annot, read_label, read_morph_data, read_geometry

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
MRI_DIR    = os.path.join(ROOT, "abide_mri_brain")
DESC_DIR   = os.path.join(ROOT, "abide_native_desc")
LGI_DIR    = os.path.join(ROOT, "abide_freesurfer6_lgi")
LABEL_DIR  = os.path.join(ROOT, "abide_label")
PHENO_CSV  = os.path.join(ROOT, "features", "Phenotypic_V1_0b.csv")
OUT_DIR    = os.path.join(ROOT, "features")

N_WORKERS  = 5          # 6 physical cores, one left for the OS
LGI_MIN, LGI_MAX = 1.0, 6.0     # plan Stage 3 plausibility bounds
Z_ABS_MAX  = 5.0                # plan Stage 3 outlier gate

# ─── ASEG LABEL MAP (verbatim from Paper 1, for baseline comparability) ──────
ASEG_LABELS = {
    4:   "L_LateralVentricle",  5:   "L_InfLatVentricle",
    7:   "L_Cerebellum_WM",     8:   "L_Cerebellum_Cortex",
    10:  "L_Thalamus",          11:  "L_Caudate",
    12:  "L_Putamen",           13:  "L_Pallidum",
    17:  "L_Hippocampus",       18:  "L_Amygdala",
    26:  "L_Accumbens",         28:  "L_VentralDC",
    43:  "R_LateralVentricle",  44:  "R_InfLatVentricle",
    46:  "R_Cerebellum_WM",     47:  "R_Cerebellum_Cortex",
    49:  "R_Thalamus",          50:  "R_Caudate",
    51:  "R_Putamen",           52:  "R_Pallidum",
    53:  "R_Hippocampus",       54:  "R_Amygdala",
    58:  "R_Accumbens",         60:  "R_VentralDC",
    16:  "Brainstem",
    251: "CC_Posterior",        252: "CC_Mid_Posterior",
    253: "CC_Central",          254: "CC_Mid_Anterior",
    255: "CC_Anterior",
    2:   "L_CerebralWM",        41:  "R_CerebralWM",
    14:  "3rdVentricle",        15:  "4thVentricle",
    24:  "CSF",
}

# The 28 subcortical graph nodes. Ventricles/CSF are measured (for TIV and QC)
# but excluded from features, exactly as in Paper 1.
FEATURE_REGIONS = [
    "L_Cerebellum_WM",  "L_Cerebellum_Cortex", "L_Thalamus",     "L_Caudate",
    "L_Putamen",        "L_Pallidum",          "L_Hippocampus",  "L_Amygdala",
    "L_Accumbens",      "L_VentralDC",
    "R_Cerebellum_WM",  "R_Cerebellum_Cortex", "R_Thalamus",     "R_Caudate",
    "R_Putamen",        "R_Pallidum",          "R_Hippocampus",  "R_Amygdala",
    "R_Accumbens",      "R_VentralDC",
    "Brainstem",
    "CC_Posterior",     "CC_Mid_Posterior",    "CC_Central",
    "CC_Mid_Anterior",  "CC_Anterior",
    "L_CerebralWM",     "R_CerebralWM",
]

# ─── DESIKAN-68 ──────────────────────────────────────────────────────────────
# 34 per hemisphere. 'unknown' (medial wall) and 'corpuscallosum' are dropped:
# they are not cortical parcels. Order is fixed here rather than taken from the
# annot colour table so column order is identical for every subject.
DESIKAN_34 = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]
EXCLUDED_PARCELS = {"unknown", "corpuscallosum", ""}


# ─── EXTRACTION PRIMITIVES ───────────────────────────────────────────────────
def extract_aseg_volumes(aseg_path):
    """Subcortical volumes in mm^3, plus TIV proxy.

    Uses one bincount pass over the volume instead of one boolean scan per
    label, which matters because this runs 1035 times.
    """
    img = nib.load(aseg_path)
    data = np.asanyarray(img.dataobj)
    if data.max() == 0:
        raise ValueError("empty segmentation")

    vox_vol = float(np.prod(img.header.get_zooms()[:3]))
    counts = np.bincount(data.ravel().astype(np.int64), minlength=256)

    vols = {name: float(counts[lab]) * vox_vol for lab, name in ASEG_LABELS.items()}
    vols["TIV_proxy"] = float(counts[1:].sum()) * vox_vol   # all labelled voxels
    return vols


def euler_characteristic(faces):
    """chi = V - E + F. A topologically correct FreeSurfer surface gives 2;
    anything else means the mesh is corrupt (plan Fix 36)."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e.sort(axis=1)
    n_edges = len(np.unique(e, axis=0))
    n_verts = int(faces.max()) + 1
    return n_verts - n_edges + len(faces)


def extract_hemi(sid, hemi):
    """All cortical measures for one hemisphere, aggregated to Desikan-34."""
    out = {}

    annot_p  = os.path.join(LABEL_DIR, sid, "label", f"{hemi}.aparc.annot")
    cortex_p = os.path.join(LABEL_DIR, sid, "label", f"{hemi}.cortex.label")
    thick_p  = os.path.join(DESC_DIR,  sid, "surf",  f"{hemi}.thickness")
    area_p   = os.path.join(DESC_DIR,  sid, "surf",  f"{hemi}.area")
    curv_p   = os.path.join(DESC_DIR,  sid, "surf",  f"{hemi}.curv")
    lgi_p    = os.path.join(LGI_DIR,   sid, "surf",  f"{hemi}.pial_lgi")
    pial_p   = os.path.join(LGI_DIR,   sid, "surf",  f"{hemi}.pial")

    labels, _, names = read_annot(annot_p)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    thickness = read_morph_data(thick_p)
    area      = read_morph_data(area_p)

    n_vert = len(thickness)
    if len(labels) != n_vert or len(area) != n_vert:
        raise ValueError(f"{hemi}: vertex count mismatch across files")

    # curv feeds only the quality index, never a model feature, so a corrupt
    # curv file degrades the subject's QC metadata rather than dropping them.
    try:
        curv = read_morph_data(curv_p)
        if len(curv) != n_vert:
            raise ValueError(f"curv has {len(curv)} vertices, expected {n_vert}")
        out[f"{hemi}_curv_ok"] = 1
    except Exception:
        curv = None
        out[f"{hemi}_curv_ok"] = 0

    # Medial wall / non-cortex mask
    cortex_idx = read_label(cortex_p)
    cortex_mask = np.zeros(n_vert, dtype=bool)
    cortex_mask[cortex_idx] = True

    # lGI is the one channel that legitimately fails for some subjects
    if os.path.exists(lgi_p):
        lgi = read_morph_data(lgi_p)
        if len(lgi) != n_vert:
            raise ValueError(f"{hemi}: lgi vertex count mismatch")
        out[f"{hemi}_lgi_missing"] = 0
    else:
        lgi = None
        out[f"{hemi}_lgi_missing"] = 1

    # name -> annot index, so a parcel missing from this subject's annot
    # produces NaN rather than a silently shifted column
    name_to_idx = {n: i for i, n in enumerate(names)}

    for parcel in DESIKAN_34:
        col = f"{hemi}_{parcel}"
        idx = name_to_idx.get(parcel)
        if idx is None:
            out[f"{col}_thickness"] = np.nan
            out[f"{col}_area"] = np.nan
            out[f"{col}_lgi"] = np.nan
            continue

        m = (labels == idx) & cortex_mask
        if not m.any():
            out[f"{col}_thickness"] = np.nan
            out[f"{col}_area"] = np.nan
            out[f"{col}_lgi"] = np.nan
            continue

        out[f"{col}_thickness"] = float(thickness[m].mean())   # mm, FreeSurfer ThickAvg convention
        out[f"{col}_area"]      = float(area[m].sum())         # mm^2, white surface
        out[f"{col}_lgi"]       = float(lgi[m].mean()) if lgi is not None else np.nan

    # ── QC measurements ──
    cm = cortex_mask
    out[f"{hemi}_n_vertices"]     = int(n_vert)
    out[f"{hemi}_n_cortex_vert"]  = int(cm.sum())
    out[f"{hemi}_curv_roughness"] = float(np.abs(curv[cm]).mean()) if curv is not None else np.nan
    out[f"{hemi}_curv_std"]       = float(curv[cm].std()) if curv is not None else np.nan
    out[f"{hemi}_mean_thickness"] = float(thickness[cm].mean())
    out[f"{hemi}_total_area"]     = float(area[cm].sum())
    if lgi is not None:
        lc = lgi[cm]
        out[f"{hemi}_mean_lgi"] = float(lc.mean())
        out[f"{hemi}_lgi_min"]  = float(lc.min())
        out[f"{hemi}_lgi_max"]  = float(lc.max())
    else:
        out[f"{hemi}_mean_lgi"] = np.nan
        out[f"{hemi}_lgi_min"]  = np.nan
        out[f"{hemi}_lgi_max"]  = np.nan

    _, faces = read_geometry(pial_p)
    out[f"{hemi}_euler"] = int(euler_characteristic(faces))

    return out


def process_subject(sid):
    """One subject -> one flat feature dict. Returns (sid, row, error)."""
    try:
        row = {"folder_id": sid, "SUB_ID": int(sid.split("_")[-1])}
        row.update(extract_aseg_volumes(os.path.join(MRI_DIR, sid, "mri", "aseg.mgz")))
        for hemi in ("lh", "rh"):
            row.update(extract_hemi(sid, hemi))
        return sid, row, None
    except Exception as exc:
        return sid, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


# ─── COHORT PASS ─────────────────────────────────────────────────────────────
def cortical_feature_columns():
    cols = []
    for measure in ("thickness", "area", "lgi"):
        for hemi in ("lh", "rh"):
            for parcel in DESIKAN_34:
                cols.append(f"{hemi}_{parcel}_{measure}")
    return cols


def apply_cohort_qc(df):
    """Gates from plan Stage 3. Measures and flags; does not drop."""
    feat_cols = FEATURE_REGIONS + cortical_feature_columns()

    # Quality index (Fix 8-replaced): curvature roughness + vertex-count deviation
    df["curv_roughness"] = df[["lh_curv_roughness", "rh_curv_roughness"]].mean(axis=1)
    df["n_vertices_total"] = df["lh_n_vertices"] + df["rh_n_vertices"]
    med_v = df["n_vertices_total"].median()
    df["vertex_count_deviation"] = (df["n_vertices_total"] - med_v).abs() / med_v

    rough_z = (df["curv_roughness"] - df["curv_roughness"].mean()) / df["curv_roughness"].std()
    vdev_z  = (df["vertex_count_deviation"] - df["vertex_count_deviation"].mean()) / df["vertex_count_deviation"].std()
    df["quality_index"] = rough_z + vdev_z

    # Gate 1: TIV bounds — cohort median +/- 3 IQR
    q1, q3 = df["TIV_proxy"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["qc_tiv_ok"] = df["TIV_proxy"].between(q1 - 3 * iqr, q3 + 3 * iqr)

    # Gate 2: no feature region may be zero
    df["qc_no_zero_region"] = (df[FEATURE_REGIONS] > 0).all(axis=1)

    # Gate 3: |z| <= 5 on every feature (NaN-tolerant: lGI-failed subjects are
    # caught by Gate 4, not here)
    z = (df[feat_cols] - df[feat_cols].mean()) / df[feat_cols].std()
    df["max_abs_z"] = z.abs().max(axis=1)
    df["qc_no_extreme_z"] = ~(z.abs() > Z_ABS_MAX).any(axis=1)

    # Gate 4: lGI present and physiologically plausible.
    # The bound is applied to the region-aggregated lGI values (the 68 node
    # features that actually enter the model), NOT to per-vertex extremes:
    # ~20% of healthy subjects exceed lGI 6.0 at their single most gyrified
    # vertex, which is normal biology rather than a reconstruction failure.
    # Per-vertex ?h_lgi_min/max are retained as diagnostics only.
    df["lgi_missing"] = (df["lh_lgi_missing"] == 1) | (df["rh_lgi_missing"] == 1)
    lgi_region_cols = [c for c in df.columns
                       if c.endswith("_lgi") and c not in ("lh_mean_lgi", "rh_mean_lgi")]
    implausible = ((df[lgi_region_cols] < LGI_MIN) | (df[lgi_region_cols] > LGI_MAX)).any(axis=1)
    df["qc_lgi_ok"] = (~df["lgi_missing"]) & (~implausible)

    # Gate 5: Euler characteristic == 2 on both pial surfaces
    df["qc_euler_ok"] = (df["lh_euler"] == 2) & (df["rh_euler"] == 2)

    # Gate 6: no missing model feature. A parcel can end up with zero cortical
    # vertices when the annot is locally wrong, which yields NaN. Such subjects
    # happen to also trip Gate 3 (the absorbed vertices make neighbouring
    # parcels extreme), but that is incidental — gate missingness directly so a
    # NaN can never reach the model.
    df["n_missing_features"] = df[feat_cols].isna().sum(axis=1)
    df["qc_no_missing_feature"] = df["n_missing_features"] == 0

    gates = ["qc_tiv_ok", "qc_no_zero_region", "qc_no_extreme_z", "qc_lgi_ok",
             "qc_euler_ok", "qc_no_missing_feature"]
    df["qc_pass"] = df[gates].all(axis=1)
    return df, gates


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subjects = sorted(d for d in os.listdir(LABEL_DIR)
                      if os.path.isdir(os.path.join(LABEL_DIR, d)))
    print(f"Subjects found: {len(subjects)}")
    print(f"Workers: {N_WORKERS}\n")

    rows, failures = [], []
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for i, (sid, row, err) in enumerate(pool.imap_unordered(process_subject, subjects, chunksize=4), 1):
            if err is None:
                rows.append(row)
            else:
                failures.append({"folder_id": sid, "error": err.splitlines()[0]})
                print(f"  FAILED {sid}: {err.splitlines()[0]}")
            if i % 50 == 0 or i == len(subjects):
                el = time.time() - t0
                eta = el / i * (len(subjects) - i)
                print(f"  {i}/{len(subjects)}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)

    df = pd.DataFrame(rows).sort_values("SUB_ID").reset_index(drop=True)
    print(f"\nExtracted {len(df)} subjects in {(time.time()-t0)/60:.1f} min")

    # ── Phenotype join ──
    # SITE_ID must come from phenotype: folder prefixes collapse LEUVEN_1/_2,
    # UCLA_1/_2 and UM_1/_2, which would silently merge LOSO folds.
    pheno = pd.read_csv(PHENO_CSV, usecols=[
        "SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ", "AGE_AT_MPRAGE",
    ])
    pheno["label"] = pheno["DX_GROUP"].map({1: 1, 2: 0})   # 1=ASD, 0=TD
    df = df.merge(pheno, on="SUB_ID", how="left", validate="one_to_one")
    unmatched = int(df["SITE_ID"].isna().sum())
    if unmatched:
        raise RuntimeError(f"{unmatched} subjects failed the phenotype join")

    df, gates = apply_cohort_qc(df)

    # ── Write ──
    out_csv = os.path.join(OUT_DIR, "abide_features_raw.csv")
    df.to_csv(out_csv, index=False)

    cort = cortical_feature_columns()
    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write(f"# subcortical nodes ({len(FEATURE_REGIONS)}) — feature: volume_mm3\n")
        f.write("\n".join(FEATURE_REGIONS) + "\n\n")
        for measure in ("thickness", "area", "lgi"):
            sel = [c for c in cort if c.endswith(f"_{measure}")]
            f.write(f"# cortical nodes ({len(sel)}) — feature: {measure}\n")
            f.write("\n".join(sel) + "\n\n")

    with open(os.path.join(OUT_DIR, "qc_report.txt"), "w") as f:
        f.write("ABIDE I — HeteroGNN feature extraction QC report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Subject folders            : {len(subjects)}\n")
        f.write(f"Extraction failures        : {len(failures)}\n")
        f.write(f"Extracted + joined         : {len(df)}\n\n")
        f.write("Gate results (subjects failing each gate, non-exclusive):\n")
        for g in gates:
            f.write(f"  {g:22s} fail = {int((~df[g]).sum()):4d}\n")
        f.write("\nNon-blocking data issues:\n")
        f.write(f"  lGI absent (>=1 hemi)     : {int(df['lgi_missing'].sum())}\n")
        f.write(f"  curv corrupt (>=1 hemi)   : {int(((df['lh_curv_ok']==0)|(df['rh_curv_ok']==0)).sum())}"
                "   (quality index falls back to the intact hemisphere)\n")
        f.write(f"\nqc_pass = True             : {int(df['qc_pass'].sum())}\n")
        f.write(f"qc_pass = False            : {int((~df['qc_pass']).sum())}\n\n")
        f.write("Cohort of qc_pass subjects:\n")
        p = df[df["qc_pass"]]
        f.write(f"  ASD / TD                 : {int((p['label']==1).sum())} / {int((p['label']==0).sum())}\n")
        f.write(f"  Male / Female            : {int((p['SEX']==1).sum())} / {int((p['SEX']==2).sum())}\n")
        f.write(f"  Sites                    : {p['SITE_ID'].nunique()}\n")
        f.write(f"  Age  mean/min/max        : {p['AGE_AT_SCAN'].mean():.1f} / {p['AGE_AT_SCAN'].min():.1f} / {p['AGE_AT_SCAN'].max():.1f}\n")
        f.write("\nPer-site counts (qc_pass):\n")
        for site, n in p["SITE_ID"].value_counts().sort_index().items():
            f.write(f"  {site:10s} {n:4d}\n")

    pd.DataFrame(failures or [{"folder_id": "", "error": ""}]).to_csv(
        os.path.join(OUT_DIR, "extraction_log.csv"), index=False)

    print(f"Wrote {out_csv}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    print(f"qc_pass: {int(df['qc_pass'].sum())} / {len(df)}")


if __name__ == "__main__":
    main()
