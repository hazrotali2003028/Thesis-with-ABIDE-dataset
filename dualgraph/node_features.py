"""
Physiologically-grounded functional node features (plan v2.1 section 4.2 upgrade).

Replaces the naive [mean_fc, std_fc] node summaries with 4 features that are
physiologically interpretable, capture distinct signal properties, and are more
resilient to the head-motion confound than raw connectivity strength:

  0. fALFF   -- fractional amplitude of low-frequency fluctuations (0.01-0.08 Hz),
                a normalised spectral measure; the fractional form divides out
                total-band power, which suppresses broadband motion spikes.
  1. Hurst   -- long-range temporal dependence / fractal self-similarity of the
                BOLD signal (rescaled-range). Non-linear complexity, orthogonal to
                spectral amplitude.
  2. wCC     -- weighted clustering coefficient (local segregation) from the FC.
  3. PC      -- participation coefficient (cross-network integration) from the FC,
                against a Yeo-7 + subcortical + cerebellum module partition.

fALFF/Hurst come from the [T,111] time-series; wCC/PC from the [111,111] matrix,
so the four are mathematically distinct constructions (2 temporal, 2 topological).

Negative-weight policy (explicit): the Fisher-z matrix is inverted to correlation
with tanh to bound weights in [-1,1], then the Rubinov-Sporns SIGNED formulations
are used -- clustering_coef_wu_sign (net signed clustering) and
participation_coef_sign, from which we take Ppos (participation over positive
edges; anticorrelation communities are not well established). Self-loops zeroed.

Install:
    pip install bctpy nolds antropy scipy numpy

Public API:
    extract_node_features(ts: np.ndarray, fc_z: np.ndarray, tr: float) -> np.ndarray
        ts    : [T, 111] float  ROI time-series (post truncation + 5 dead-ROI drop)
        fc_z  : [111, 111] float  Ledoit-Wolf Fisher-z connectivity
        tr    : float           repetition time in seconds (site-specific)
        returns [111, 4] float32, columns = [fALFF, Hurst, wCC, PC], NaN/Inf-safe
"""

from __future__ import annotations

import numpy as np
from scipy import signal

import bct
import nolds

N_NODES: int = 111
DEAD_ROI_0BASED: tuple[int, ...] = (86, 100, 101, 106, 107)   # dropped from AAL-116
LOW_BAND_HZ: tuple[float, float] = (0.01, 0.08)


# ---------------------------------------------------------------- module partition
def _aal116_partition() -> np.ndarray:
    """Length-116 module labels (1..9): Yeo-7 cortical + subcortical + cerebellum.

    Assigned by the fixed AAL-116 index order. Approximate intrinsic-network
    membership (a documented anatomical mock, per the plan's allowance), NOT a
    voxelwise Yeo overlap. Modules: 1 Visual, 2 SomMot, 3 DorsAttn, 4 VentAttn,
    5 Limbic, 6 Control, 7 Default, 8 Subcortical, 9 Cerebellum.
    """
    ci = np.zeros(116, dtype=np.int64)
    ranges = {
        2: [(0, 2), (16, 20), (56, 58), (68, 70), (78, 80)],          # SomMot
        6: [(2, 4), (6, 8), (12, 14)],                                # Control (FPN)
        5: [(4, 6), (8, 10), (14, 16), (20, 22), (26, 28),            # Limbic
            (36, 42), (82, 84), (86, 88)],
        4: [(10, 12), (28, 30), (30, 34), (62, 64)],                  # VentAttn/Salience
        7: [(22, 26), (34, 36), (64, 68), (80, 82), (84, 86), (88, 90)],  # Default
        1: [(42, 56)],                                                # Visual
        3: [(58, 62)],                                                # DorsAttn
        8: [(70, 78)],                                                # Subcortical
        9: [(90, 116)],                                               # Cerebellum
    }
    for mod, spans in ranges.items():
        for lo, hi in spans:
            ci[lo:hi] = mod
    assert (ci > 0).all(), "unassigned AAL node"
    return ci


_CI_116 = _aal116_partition()
YEO_CI_111: np.ndarray = np.delete(_CI_116, list(DEAD_ROI_0BASED))
assert YEO_CI_111.shape[0] == N_NODES


# ------------------------------------------------------------------------ features
def _falff(ts: np.ndarray, tr: float) -> np.ndarray:
    """[T,R] -> [R] fractional ALFF over LOW_BAND_HZ."""
    fs = 1.0 / float(tr)
    freqs, psd = signal.periodogram(ts, fs=fs, axis=0, detrend="constant")
    amp = np.sqrt(np.clip(psd, 0.0, None))                 # amplitude spectrum
    band = (freqs >= LOW_BAND_HZ[0]) & (freqs <= LOW_BAND_HZ[1])
    total = amp.sum(axis=0)
    total[total == 0] = 1.0
    return amp[band, :].sum(axis=0) / total


def _hurst(ts: np.ndarray) -> np.ndarray:
    """[T,R] -> [R] rescaled-range Hurst exponent per ROI (nolds), robust to
    short/degenerate series."""
    T, R = ts.shape
    out = np.full(R, 0.5, dtype=np.float64)                # 0.5 = uninformative default
    for j in range(R):
        x = ts[:, j]
        if x.std() < 1e-12:
            continue
        try:
            out[j] = nolds.hurst_rs(x)
        except Exception:
            out[j] = 0.5
    return out


def _graph_metrics(fc_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[R,R] Fisher-z -> (weighted clustering, participation) per node."""
    W = np.tanh(fc_z).astype(np.float64)                   # back to correlation [-1,1]
    np.fill_diagonal(W, 0.0)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    cpos, _cneg = bct.clustering_coef_wu_sign(W)           # (positive, negative) clustering
    ppos, _pneg = bct.participation_coef_sign(W, YEO_CI_111)
    return np.asarray(cpos), np.asarray(ppos)


def extract_robust_features(fc_z: np.ndarray) -> np.ndarray:
    """[111,111] Fisher-z -> [111,4] robust FC-graph 'nodal role' features:
    [positive strength, eigenvector centrality, within-module degree z, PC].

    All FC-derived (no T/TR dependence), stable at 111 nodes. Weights bounded via
    tanh; strength uses positive edges, centrality/within-module-z use |r|
    (non-negativity required), PC reuses the signed Ppos (identical to physio4's
    PC, so the comparison isolates the other three)."""
    fc_z = np.asarray(fc_z, dtype=np.float64)
    if fc_z.shape != (N_NODES, N_NODES):
        raise ValueError(f"fc_z must be [{N_NODES}, {N_NODES}], got {fc_z.shape}")
    W = np.tanh(fc_z)
    np.fill_diagonal(W, 0.0)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    Wpos = np.where(W > 0, W, 0.0)
    Wabs = np.abs(W)

    strength = Wpos.sum(axis=1)
    eigcen = bct.eigenvector_centrality_und(Wabs)
    wmz = bct.module_degree_zscore(Wabs, YEO_CI_111, flag=0)
    _cpos = None
    ppos, _pneg = bct.participation_coef_sign(W, YEO_CI_111)

    feat = np.column_stack([strength, np.asarray(eigcen), np.asarray(wmz),
                            np.asarray(ppos)]).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def extract_node_features(ts: np.ndarray, fc_z: np.ndarray, tr: float) -> np.ndarray:
    """Extract the [111, 4] node-feature matrix. See module docstring."""
    ts = np.asarray(ts, dtype=np.float64)
    fc_z = np.asarray(fc_z, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[1] != N_NODES:
        raise ValueError(f"ts must be [T, {N_NODES}], got {ts.shape}")
    if fc_z.shape != (N_NODES, N_NODES):
        raise ValueError(f"fc_z must be [{N_NODES}, {N_NODES}], got {fc_z.shape}")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError(f"tr must be positive finite, got {tr}")

    falff = _falff(ts, tr)
    hurst = _hurst(ts)
    wcc, pc = _graph_metrics(fc_z)

    feat = np.column_stack([falff, hurst, wcc, pc]).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":       # quick self-test on random data
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((116, N_NODES))
    fc = np.arctanh(np.clip(np.corrcoef(ts.T), -0.999, 0.999))
    f = extract_node_features(ts, fc, tr=2.0)
    print("shape", f.shape, "finite", np.isfinite(f).all())
    print("col means", f.mean(0).round(3))
