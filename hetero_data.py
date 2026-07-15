"""
Heterogeneous graph construction (plan Stages 4-6).

Everything that could leak is fitted inside build_fold() on the training split
only: ComBat, the feature scaler, and the covariation adjacency. The held-out
site sees forward application exclusively.

Node design (plan Fix 11 — no zero padding; each anatomy keeps its own space):
    cortical    68 Desikan nodes, x = [thickness, area, lGI, cx, cy, cz]  -> 6
    subcortical 28 aseg nodes,    x = [volume, cx, cy, cz]                -> 4

Relations:
    (cortical,    covaries, cortical)
    (subcortical, covaries, subcortical)
    (cortical,    covaries, subcortical)  + reverse, so messages flow both ways

Edges are structural covariance: Pearson correlation ACROSS TRAINING SUBJECTS
between each node's representative measure (thickness for cortical, volume for
subcortical), thresholded at |r| > tau. Homotopic left/right pairs are unioned
into the within-type relations ("asymmetry edges" in the plan).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from extract_features import FEATURE_REGIONS, DESIKAN_34
from combat import combat_fit, combat_apply_train, combat_apply_unseen

CORTICAL_NODES = [f"{h}_{p}" for h in ("lh", "rh") for p in DESIKAN_34]   # 68
SUBCORTICAL_NODES = list(FEATURE_REGIONS)                                # 28
CORTICAL_MEASURES = ("thickness", "area", "lgi")

CT = "cortical"
SC = "subcortical"
R_CC = (CT, "covaries", CT)
R_SS = (SC, "covaries", SC)
R_CS = (CT, "covaries", SC)
R_SC = (SC, "rev_covaries", CT)


def feature_matrix(df):
    """(n, 232) in a fixed, documented column order."""
    cols = list(SUBCORTICAL_NODES)
    for node in CORTICAL_NODES:
        cols += [f"{node}_{m}" for m in CORTICAL_MEASURES]
    return df[cols].to_numpy(dtype=np.float64), cols


def _corr_edges(M, tau):
    """|Pearson| > tau over columns of M (subjects x nodes). No self loops."""
    C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    return np.abs(C) > tau, C


def _homotopic_pairs(nodes):
    """Indices of left/right homologues within one node list."""
    idx = {n: i for i, n in enumerate(nodes)}
    pairs = []
    for n, i in idx.items():
        if n.startswith("lh_"):
            j = idx.get("rh_" + n[3:])
        elif n.startswith("L_"):
            j = idx.get("R_" + n[2:])
        else:
            continue
        if j is not None:
            pairs.append((i, j))
    return pairs


def _to_edge_index(mask, weights=None):
    src, dst = np.nonzero(mask)
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    if weights is None:
        return ei, None
    w = torch.tensor(weights[src, dst], dtype=torch.float32).unsqueeze(1)
    return ei, w


def harmonize_and_scale(df, train_idx, test_idx, use_combat=True):
    """Stage 4 + scaler for one split, fitted on the training rows only.

    Shared by the HeteroGNN and every baseline so that the ONLY difference
    between them is the model. If the baselines saw different preprocessing,
    "the graph adds value" would be uninterpretable.

    Returns:
        (Ztr, Zte, ytr, yte, cols)
    """
    Y, cols = feature_matrix(df)
    dx = df["label"].to_numpy(dtype=float)
    age = df["AGE_AT_SCAN"].to_numpy(dtype=float)
    sex = df["SEX"].to_numpy(dtype=float)
    site = df["SITE_ID"].to_numpy()

    # FIQ is imputed at the modelling stage (plan Fix 9); it is not a node
    # feature, so it does not enter the graph here.
    X_protect = np.column_stack([dx, age, sex])
    NOLABEL_IDX = [1, 2]                      # age, sex  (dx is column 0)

    Ytr, Yte = Y[train_idx], Y[test_idx]

    if use_combat:
        est = combat_fit(Ytr, site[train_idx], X_protect[train_idx], NOLABEL_IDX)
        Htr = combat_apply_train(Ytr, site[train_idx], X_protect[train_idx], est)
        Hte = combat_apply_unseen(Yte, X_protect[test_idx], est)
    else:
        Htr, Hte = Ytr.copy(), Yte.copy()

    mu, sd = Htr.mean(axis=0), Htr.std(axis=0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Htr - mu) / sd, (Hte - mu) / sd
    return (Ztr, Zte,
            df.loc[train_idx, "label"].to_numpy(),
            df.loc[test_idx, "label"].to_numpy(),
            cols)


def build_fold(df, train_idx, test_idx, coords, tau=0.3, use_combat=True):
    """Harmonize, scale and wire one LOSO fold.

    Args:
        df:        feature table (already restricted to qc_pass subjects)
        train_idx: boolean array over df rows
        test_idx:  boolean array over df rows
        coords:    node_coords.csv as a DataFrame
        tau:       covariation edge threshold

    Returns:
        (train_graphs, test_graphs, info)
    """
    Ztr, Zte, ytr, yte, cols = harmonize_and_scale(df, train_idx, test_idx, use_combat)

    # ── Stage 6: covariation adjacency from TRAINING subjects only ──
    ci = {c: i for i, c in enumerate(cols)}
    sub_cols = [ci[n] for n in SUBCORTICAL_NODES]
    cort_cols = [ci[f"{n}_thickness"] for n in CORTICAL_NODES]

    cc_mask, cc_w = _corr_edges(Ztr[:, cort_cols], tau)
    ss_mask, ss_w = _corr_edges(Ztr[:, sub_cols], tau)

    # between types: correlate cortical thickness against subcortical volume
    Ccs = np.corrcoef(Ztr[:, cort_cols + sub_cols], rowvar=False)
    Ccs = np.nan_to_num(Ccs, nan=0.0)
    n_ct = len(CORTICAL_NODES)
    cs_w = Ccs[:n_ct, n_ct:]
    cs_mask = np.abs(cs_w) > tau

    # homotopic ("asymmetry") edges, unioned into within-type relations
    for i, j in _homotopic_pairs(CORTICAL_NODES):
        cc_mask[i, j] = cc_mask[j, i] = True
    for i, j in _homotopic_pairs(SUBCORTICAL_NODES):
        ss_mask[i, j] = ss_mask[j, i] = True

    ei_cc, ew_cc = _to_edge_index(cc_mask, cc_w)
    ei_ss, ew_ss = _to_edge_index(ss_mask, ss_w)
    ei_cs, ew_cs = _to_edge_index(cs_mask, cs_w)
    ei_sc = ei_cs.flip(0)
    ew_sc = ew_cs

    # ── node coordinates (constant across subjects), standardized once ──
    cmap = coords.set_index("node")
    P_ct = cmap.loc[CORTICAL_NODES, ["x", "y", "z"]].to_numpy(dtype=np.float32)
    P_sc = cmap.loc[SUBCORTICAL_NODES, ["x", "y", "z"]].to_numpy(dtype=np.float32)
    allp = np.vstack([P_ct, P_sc])
    P_ct = (P_ct - allp.mean(0)) / allp.std(0)
    P_sc = (P_sc - allp.mean(0)) / allp.std(0)
    P_ct_t = torch.tensor(P_ct)
    P_sc_t = torch.tensor(P_sc)

    def make(Z, labels):
        out = []
        ct_feat = np.stack(
            [Z[:, [ci[f"{n}_{m}"] for n in CORTICAL_NODES]] for m in CORTICAL_MEASURES],
            axis=-1)                                   # (n, 68, 3)
        sc_feat = Z[:, sub_cols][:, :, None]           # (n, 28, 1)
        for k in range(len(Z)):
            d = HeteroData()
            d[CT].x = torch.cat(
                [torch.tensor(ct_feat[k], dtype=torch.float32), P_ct_t], dim=1)
            d[SC].x = torch.cat(
                [torch.tensor(sc_feat[k], dtype=torch.float32), P_sc_t], dim=1)
            d[R_CC].edge_index, d[R_CC].edge_attr = ei_cc, ew_cc
            d[R_SS].edge_index, d[R_SS].edge_attr = ei_ss, ew_ss
            d[R_CS].edge_index, d[R_CS].edge_attr = ei_cs, ew_cs
            d[R_SC].edge_index, d[R_SC].edge_attr = ei_sc, ew_sc
            d.y = torch.tensor([labels[k]], dtype=torch.long)
            out.append(d)
        return out

    train_graphs = make(Ztr, ytr)
    test_graphs = make(Zte, yte)

    info = {
        "n_train": int(train_idx.sum()),
        "n_test": int(test_idx.sum()),
        "edges_cc": int(cc_mask.sum()),
        "edges_ss": int(ss_mask.sum()),
        "edges_cs": int(cs_mask.sum()),
        "density_cc": float(cc_mask.mean()),
        "density_ss": float(ss_mask.mean()),
        "density_cs": float(cs_mask.mean()),
    }
    return train_graphs, test_graphs, info


def site_cv_reduction(df, harmonized, cols):
    """Between-site variance of site means, averaged over features (plan Stage 4
    asks for the CV reduction to be reported)."""
    site = df["SITE_ID"].to_numpy()
    v = []
    for j in range(harmonized.shape[1]):
        means = [harmonized[site == s, j].mean() for s in np.unique(site)]
        v.append(np.var(means))
    return float(np.mean(v))
