# Cortical-Expanded Heterogeneous GNN for Multi-Site ASD Classification (ABIDE I)

Structural-only ASD vs TD classification on ABIDE I, using precomputed FreeSurfer 6
outputs read with `nibabel` (no FreeSurfer install required).

**Headline result: the heterogeneous graph does not beat a linear model.** Details below.

## Data

Not included in this repository. The raw archives (~33 GB) and the individual-level
feature table are covered by the ABIDE data use agreement and are gitignored. To
reproduce, obtain ABIDE I from
[the Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/)
and run `extract_features.py`, which regenerates the feature table in ~13 minutes.

Cohort after QC: **979 subjects** (466 ASD / 513 TD), 20 sites, 232 features per subject.

## Results

All numbers are leave-one-site-out (LOSO) over 20 sites, with ComBat and the feature
scaler fitted on training sites only. The GNN additionally uses nested LOSO
(hyperparameters chosen in an inner loop that never sees the test site).

| model | LOSO AUC |
|---|---|
| svm_rbf | **0.6037 ± 0.0889** |
| rf | 0.5969 ± 0.0885 |
| xgboost | 0.5860 ± 0.0875 |
| logreg | 0.5794 ± 0.0897 |
| mlp (no graph) | 0.5720 ± 0.0584 |
| svm_linear | 0.5544 ± 0.0892 |
| **HeteroGNN** | **0.5557 ± 0.0868** |
| dummy (majority class) | 0.5000 |

The HeteroGNN loses to every non-trivial baseline and, after Holm correction, does not
significantly beat the majority-class dummy (p = 0.082).

### Three findings

**1. The graph contributes nothing.** Sweeping the covariation threshold τ from 0.3 to 0.9
changes cortical graph density from 78.5% to 1.5% and moves AUC by less than 0.02. The
adjacency is computed *across* subjects, so it is byte-identical for every subject — it
carries zero subject-specific information and the GNN degenerates to an MLP with a fixed
mixing matrix. This independently reproduces Paper 1's GAT ≈ MLP (p = 0.984).

**2. Epoch selection on the test fold inflates AUC by +0.10.** Measured directly over all
20 sites × 3 seeds: the same model, data and training run, differing only in which epoch
is reported.

| epoch chosen on | AUC |
|---|---|
| a held-out validation split (honest) | 0.5497 ± 0.081 |
| the test site itself | 0.6502 ± 0.071 |
| **inflation** | **+0.1004 ± 0.063** (Wilcoxon p = 1.9e-06) |

The honest figure (0.5497) independently reproduces the nested-LOSO result (0.5557) via a
separate code path.

Inflation ranges +0.025 to +0.243 and is driven by **how poorly the model genuinely
performs**, not by site size:

| predictor of inflation | r | p |
|---|---|---|
| honest AUC | **−0.541** | ~0.014 |
| site size | −0.374 | 0.104 (n.s.) |

Low true skill leaves more headroom for lucky noise. The clearest case is UM_2: honest AUC
**0.389** — worse than a coin flip — reported as **0.632** under Paper 1's rule. Compare
LEUVEN_2, genuinely decent at 0.662, which gains only +0.025.

**Comparison to Paper 1.** Paper 1 reported GAT v3 at 0.635 ± 0.052 over these same 20
sites using this rule. The HeteroGNN under the same rule scores **0.6502** — with 60 epochs
against Paper 1's 200, so with fewer chances to maximise over. This suggests the newer
architecture is at least as good on Paper 1's own terms, though without Paper 1's per-site
numbers the +0.015 margin cannot be tested for significance.

Both figures are protocol artefacts, not out-of-sample estimates. Under honest validation
both architectures land near 0.55, below the 0.6037 svm_rbf baseline. See
`check_paper1_protocol.py`.

**3. GraphNorm destroys the signal under mean pooling.** GraphNorm centres each channel
across the nodes of a graph; `global_mean_pool` then averages over those same nodes, so the
readout collapses toward a learned constant. Measured across-subject SD of the pooled graph
vector: **0.024 (GraphNorm) vs 0.199 (BatchNorm)**. Switching to BatchNorm recovered +0.078 AUC.

### Validation

- **ComBat** matches the reference `neuroCombat` to 1.087e-07 while additionally supporting
  forward application to an unseen site (`neuroCombat` refuses this outright). Permuting
  test-site labels changes harmonized output by exactly 0.0. See `test_combat.py`.
- **ComBat is not removing ASD signal** (Paper 1's "Suspect 4"): harmonized ≥ raw for every
  model, and cross-site AUC spread drops (svm_rbf 0.111 → 0.089). See `check_harmonization.py`.
- **Temperature scaling** recovers known miscalibration (k=3 → T=3.02) and leaves AUC
  bit-identical. See `test_calibration.py`.

## Pipeline

```
extract_features.py      FreeSurfer outputs -> 232 features + QC   
build_coords.py          96 node centroids -> features/node_coords.csv
combat.py                ComBat: train-only fit, unseen-site forward apply
hetero_data.py           harmonization + scaler + covariation graph (all train-only)
hetero_gnn.py            HeteroConv/GATv2 + BatchNorm + temperature scaling
train_baselines.py       7 baselines through identical folds       
train_nested_loso.py     nested LOSO, resumable                    
compare_models.py        paired Wilcoxon + Holm, GNN vs baselines
check_harmonization.py   A/B: harmonized vs raw features
check_paper1_protocol.py measures epoch-selection inflation
```

Tests: `test_combat.py`, `test_calibration.py`.

### Reproduce

```bash
pip install nibabel numpy pandas scikit-learn scipy xgboost neuroCombat
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric

python extract_features.py      # writes features/
python build_coords.py 150
python train_baselines.py       # writes results/baseline_*
python train_nested_loso.py     # writes results/nested_loso_results.csv
python compare_models.py        # writes results/model_comparison*
```

## Notes on coordinates

`features/node_coords.csv` holds cohort-mean centroids in FreeSurfer surface RAS (tkrRAS),
**not MNI**. True MNI would need fsaverage or per-subject `talairach.xfm`, neither of which
is in the archives. They serve as a fixed positional encoding, which only requires a
consistent frame. Do not cite them as MNI, and do not reuse them for spin tests (which need
spherical surface coordinates).

## Environment

Developed on a GTX 1650 Max-Q (4 GB), i5-11400H, 7.7 GB RAM, Windows 11. The cu118 torch
build runs on driver 511.65 via CUDA minor-version compatibility. Feature extraction streams
one subject per worker to stay within the RAM budget.
