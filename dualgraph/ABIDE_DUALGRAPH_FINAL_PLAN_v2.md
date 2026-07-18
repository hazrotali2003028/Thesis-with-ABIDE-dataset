# ABIDE Dual-Graph Cross-Attention — Execution Plan v2.1

**Status:** blockers B1–B3 RESOLVED (§1). Only unfilled item = Stage 2 numeric output (Nf, DX split, T\*). Design final; run Stage 2 to fill slots.
**v2.1 changes:** MDE corrected 0.02→0.06–0.07 (§8); PR-AUC added (§7,§8); MLP parity rungs 1a-MLP/1b-MLP (§7); inner loop → StratifiedGroupKFold by SITE\_ID (§6); T\* justified via empirical CDF (§2,§3.2).
**Supersedes:** `ABIDE\_MULTIMODAL\_HANDOFF.md` (§0, §4.1, §4.2, §4.5 are wrong — see §0 below).
**Compute target:** Kaggle (12 h/session, 30 h/wk GPU) + local machine.
**Nothing is pushed to GitHub yet. Do not push `README.md` until Stage 0 completes.**

\---

## 0\. Corrections to the prior record

These are load-bearing. Every one is a measured result from your own execution log or code, not an opinion.

|Prior claim|Status|Evidence|
|-|-|-|
|GAT v3 sMRI-only = **0.635 ± 0.052**|**INVALID**|Epoch selected on the test site. Inflation measured at **+0.129 AUC, p = 0.031**. Honest ≈ **0.51**.|
|"Two independent architectures agree graphs add nothing"|**FALSE**|Both use group-level covariation at the **same τ = 0.3**. Paper 1: `build\_adjacency(X\_train, threshold=0.3)`. HeteroGNN: `\_corr\_edges(Ztr, tau=0.3)`. One design error, run twice.|
|"Paper 1 had per-subject edges and still got \~0.51"|**UNSUPPORTED**|`compute\_subject\_edge\_weights` modulates **bilateral edges only** (\~14 pairs of \~454 directed edges ≈ 6%), and the modulator `\|vol\_L−vol\_R\|/(vol\_L+vol\_R+ε)` is a **deterministic function of the volumes already in the node features**. It injects zero new information.|
|README: "the heterogeneous graph does not beat a linear model"|**UNEARNED**|`tau` is pinned to `\[0.3]` in `HP\_GRID` and was never searched. Density 0.750 (cortical) ⇒ near-complete graph ⇒ negligible inductive bias. The null was never tested against a *structured* graph.|
|Baseline floor to beat|**CHANGED**|sMRI: `svm\_rbf = 0.6037`. fMRI: **motion-only ≈ 0.58** (derived §9.1).|

**Consequence:** the τ sweep (Stage 0) is not optional. It is the Section-2 motivation of the paper. Without it, "group-level graphs fail" is an assertion.

\---

## 1\. BLOCKING — RESOLVED

**B1/B2 confirmed.** Cortical `\[thickness, area, lGI, x, y, z]` 68×6, subcortical `\[volume, x, y, z]` 28×4. No `label`/`DX\_GROUP` leak. Matches §4.1 exactly.
**B3 resolved.** 68+28 = **96 nodes**, confirmed.

## 1.1 NEW BLOCKING — unverified numbers in §2

Prior turn's §2 table stated T-per-site and DX-per-site as *measured*. **They were not — retract.** Only confirmed: 979 sMRI QC / 884 AAL / 840 ∩ / 816 after n>20 (CMU=5, UCLA\_2=19 dropped) / OHSU n=22 on overlap. **NOT confirmed:** OHSU T=78, T≥116 rule, 794 final N, any DX split (380/460, 369/447, OHSU split) — none computed on the overlap. Only known DX split is full 979 sMRI cohort (466/513), a **different, larger set** — do not reuse.

**Action before Stage 2:** compute per-subject `T` from the 884 `.1D` files directly (`np.loadtxt(f).shape\[0]`) and per-site DX counts from `cohort\_840.csv` merge. §2 below is **PROVISIONAL** until re-run.

\---

## 2\. Cohort — PROVISIONAL (T/DX rows unverified, see §1.1)

```
1112   ABIDE PCP phenotypic rows          CONFIRMED
 979   sMRI QC pass                       CONFIRMED
 884   AAL timeseries on disk             CONFIRMED (FD<0.2 pre-filtered)
 840   sMRI ∩ fMRI                        CONFIRMED (N only; DX split NOT computed)
 816   after n>20 rule (drop CMU 5, UCLA\_2 19)  CONFIRMED → 18 sites
  Nf   after T-rule                       ← Stage 2 computes. T never measured.
```

**Site-N confirmed (Σ=816):** 39/55/36/42/34/169/32/43/24/48/24/33/59/28/32/68/28. **DX split and T-column: NOT computed — no fabricated values in this plan.**

**Stage 2 is the freeze. It computes every missing number, then writes the manifest. Run this first — its stdout fills the slots below.**

```python
import numpy as np, pandas as pd, glob, os
# ---- T per file ----
rows=\[{"FILE\_ID":os.path.basename(f).replace("\_rois\_aal.1D",""),
       "T":np.loadtxt(f).shape\[0]}
      for f in glob.glob("abide\_fmri/Outputs/cpac/nofilt\_noglobal/rois\_aal/\*.1D")]
tf=pd.DataFrame(rows)
ph=pd.read\_csv("Phenotypic\_V1\_0b\_preprocessed1.csv")          # PCP file w/ func\_mean\_fd
c=tf.merge(ph\[\["FILE\_ID","SITE\_ID","DX\_GROUP","AGE\_AT\_SCAN","SEX","FIQ","func\_mean\_fd"]],
           on="FILE\_ID",how="inner")
c=c\[c.SITE\_ID.map(c.SITE\_ID.value\_counts())>20]               # n>20 → 816
print("T by site:\\n", c.groupby("SITE\_ID")\["T"].agg(\["count","min","median","max"]))
print("DX by site:\\n", c.groupby("SITE\_ID")\["DX\_GROUP"].value\_counts().unstack(fill\_value=0))
print("840-overlap split:", (c.DX\_GROUP==1).sum(),"ASD /",(c.DX\_GROUP==2).sum(),"TD")
print("T percentiles:", np.percentile(c.T,\[10,15,20,25,50]))  # pick T\* from CDF
T\_RULE=116     # PLACEHOLDER — set to the 15–20th percentile printed above, not a guess
cf=c\[c.T>=T\_RULE]
print("FINAL Nf =",len(cf)," sites =",cf.SITE\_ID.nunique(),
      " split =",(cf.DX\_GROUP==1).sum(),"/",(cf.DX\_GROUP==2).sum())
cf.to\_csv("cohort\_final.csv",index=False)
```

**Fill after run (do NOT hand-guess):**
`Nf = \_\_\_\_   sites = \_\_\_\_   split = \_\_\_\_ ASD / \_\_\_\_ TD   dropped-by-T = \_\_\_\_`
Then every `\[Nf,116,116]` / `cohort\_final.csv` below resolves. If OHSU (n=22) median-T ≥ 116 it stays and Nf jumps — the earlier "794/17-site" figure was a guess and is void until this prints.

### 2.1 Confound status (measured — do not re-litigate)

|Confound|Result|Verdict|
|-|-|-|
|Site × DX|χ², p = 0.267|**Clean.** No site–label association.|
|T vs DX (subject)|r = −0.042, p = 0.234|**Clean.**|
|T vs %ASD (site, n=18)|r = −0.171, p = 0.497|**Clean.**|
|**FD vs DX (subject)**|**r = +0.138, p = 0.0001**|**CONFOUNDED.** Survives the FD<0.2 filter.|
|FD by group|ASD 0.0872 vs TD 0.0754, p = 0.0001|ASD move more in **12 of 18** sites.|

**T is orthogonal to the label ⇒ scan-length heterogeneity costs precision, not validity.** Truncation is therefore a precision/homogeneity trade, not a leak fix. It is primary because it is reviewer-proof, but full-T is a mandatory ablation (§7 A3) because truncating UM\_1 296→116 *raises* its edge SE from 0.058 to 0.094.

**FD is confounded ⇒ every motion gate in §9 is mandatory, not optional.**

### 2.2 Selection-bias disclosure (goes in the Abstract, not Limitations)

The 884 files are pre-filtered at `mean\_fd\_thresh = 0.2`. In the unfiltered phenotypic file FD reaches 1.43 and **162 subjects exceed threshold**. Exclusion is **not diagnosis-neutral** (ASD move more), so survivors skew low-motion — typically older, higher-IQ, less severe.

**Required before writing:**

* DX split of the 162 excluded.
* Do the Nf differ from the 979 on `AGE\_AT\_SCAN`, `FIQ`, `ADOS\_TOTAL`? (Mann–Whitney, report U and p.)

**The claim you are licensed to make is "low-motion ASD," not "ASD."**

Range restriction also means: **a null `r(label, FD)` inside this cohort would not license "motion is controlled"** — it would mean you cannot see it. You already measured r = +0.138 **despite** the restriction, so the true effect is larger.

\---

## 3\. Data pipeline

### 3.1 Structural (exists — reuse)

`features/abide\_features\_raw.csv` → 232-vector (28 volumes + 68×3 cortical) → `harmonize\_and\_scale()` → ComBat (**fit train rows only**) → z-score (train μ/σ) → `Ztr`.
`features/node\_coords.csv` → 96×3 MNI centroids, standardized in `build\_fold`.

**Known property, state it in the paper:** of the 96×6 + … input values, only **232 vary per subject**; the coordinate channels are group constants (positional encoding). This is intentional, not a bug.

### 3.2 Functional (new)

```
abide\_fmri/Outputs/cpac/nofilt\_noglobal/rois\_aal/\*.1D   →  ts: \[T, 116]
```

**Step F1 — load \& QC.** Reject if `ndim != 2`, `T < 116`, `R != 116`. Count zero-variance ROIs per subject; **if dead ROIs are site-systematic, that is FOV clipping and becomes fake graph structure** — log the per-site count before doing anything else.

**Step F2 — truncate.** `ts = ts\[:T\*, :]`, first `T\*` volumes, deterministic. **`T\*` is the 15–20th percentile of the empirical T distribution over the 816 n>20 subjects (Stage 2 prints the CDF), not a hand-picked 116.** State it in the paper as: *"T\* = \_\_\_ was chosen as the \_\_\_th percentile of the empirical T distribution, retaining ≥80% of subjects while equalizing sequence length."* The full-T vs T\* comparison stays as A3, now a justified rule rather than a guess.

> \*\*Limitation to state:\*\* `func\_mean\_fd` is computed by PCP over the \*\*full\*\* scan. Your analysis window is the first 116 volumes. For long-scan sites the FD covariate over-represents motion outside the analysis window. Recomputing FD on the window needs the motion-parameter files, which are \*\*not\*\* in the `rois\_aal` derivative you downloaded. Accept and disclose.

**Step F3 — connectivity.** Do **not** use `np.corrcoef`. At T=116 with 116 ROIs the sample correlation matrix is near-singular (rank ≤ 115).

```python
from nilearn.connectome import ConnectivityMeasure
cm = ConnectivityMeasure(kind="correlation", cov\_estimator=LedoitWolf(store\_precision=False))
```

Ledoit–Wolf shrinkage intensity **adapts to T**, which partially self-corrects the residual 1.61× SE gap (T=116 → SE 0.094; T=296 → 0.058).

**Step F4 — Fisher-z.** `fc\_z = arctanh(clip(fc, -0.999999, 0.999999))`, zero the diagonal.

**Step F5 — edge ComBat (ablation A4).** On the 6,670 upper-triangle Fisher-z values, `batch = SITE\_ID`, protected covariates = `\[DX\_GROUP, AGE\_AT\_SCAN, SEX, func\_mean\_fd]`, **fit on train fold only**. Your ComBat result (21% of cortical edges / 28% of subcortical edges flip with/without) proves the adjacency sits **inside** the leakage boundary — so per-fold refit is mandatory, and the same discipline applies here.

> \*\*Caveats on 1/√(T−3), do not over-trust it:\*\* TR varies across sites, so equal T ≠ equal duration or equal effective DOF — you need the ABIDE site table for that. And these are `nofilt` derivatives: the timeseries are autocorrelated, so 1/√(T−3) is a \*\*lower bound\*\* on true SE.

**Cache:** `fc\_z.npy` → `\[Nf, 116, 116]` float32 ≈ **42.7 MB**. Trivial. Compute once, version the hash.

\---

## 4\. Graph construction

### 4.1 Graph 1 — structural, HeteroData

```
x\_cort : \[68, 6]   \[thickness, area, lGI, x\_MNI, y\_MNI, z\_MNI]   ← 3 vary, 3 constant
x\_sub  : \[28, 4]   \[volume, x\_MNI, y\_MNI, z\_MNI]                 ← 1 varies, 3 constant
```

**Edges — SubjectAdaptiveGraph (replaces `\_corr\_edges`):**

```python
h = Linear(d\_in, d\_proj)(x)          # \[96, d\_proj]
h = F.normalize(h, p=2, dim=-1)
S = h @ h.T                          # \[96, 96] cosine sim, PER SUBJECT
S.fill\_diagonal\_(-inf)
edge\_index = topk(S, k=k, dim=-1)    # k per node, symmetrised
edge\_attr  = gather(S, edge\_index)   # \[E, 1]
```

`edge\_index` now varies per subject. **Verify with the same assertion that caught the bug: `std(edge\_index) across 50 subjects > 0`.** Make this a unit test, not a hope.

**`k` MUST be swept — it is the new `tau`.** `k ∈ {5, 10, 20, 30}`. At k=30 on 96 nodes density = 0.31; at k=5, 0.05. Pinning k repeats the exact error that produced density 0.750.

### 4.2 Graph 2 — functional

```
x\_func : \[116, 4]   \[mean\_fc, std\_fc, z\_age, b\_sex]
edges  : |fc\_z\_ij| > θ   OR   top-k per node
edge\_attr : fc\_z\_ij      \[E, 1]   ← MANDATORY. Without it you discard the FC values.
```

`θ` or `k\_func` swept identically. **`edge\_attr` is non-negotiable:** node features carry 232 numbers; the FC matrix carries 6,670. If the values live only in a binary topology, the GNN competes against the FC-SVM holding strictly less.

### 4.3 The information ceiling — state this in the paper

* Graph 1: `A = f(X)` ⇒ encoder output `= h(X)`. **Same information as svm\_rbf on the 232-vector.**
* Graph 2: `mean\_fc`, `std\_fc` are derivable from `fc\_z` ⇒ **same information as SVM on the 6,670-vector.**

**A GNN never has more information than a flat model on the same raw data.** The only testable advantage is **inductive bias at N≈780 train subjects**. Frame every claim that way. This is exactly CAS-GNN's one significant module (subject-adaptive graph) and their cross-attention was **p = 0.053 — not significant**.

\---

## 5\. Architecture

```
x\_cort \[B,68,6] ──Linear──┐
                          ├──> \[B,96,64] ──HeteroGNN──> S \[B,96,64]
x\_sub  \[B,28,4] ──Linear──┘

x\_func \[B,116,4] ─Linear──> \[B,116,64] ──HeteroGNN───────> F \[B,116,64]

            ┌──────── cross-attention ────────┐
Variant A:  Q=S, K=V=F  → MHA → \[B,96,64]   (sMRI queries fMRI)   ← the novelty
Variant B:  Q=F, K=V=S  → MHA → \[B,116,64]  (CAS-GNN direction)
Variant C:  both, concat                     (bidirectional)

            → mean-pool → \[B,64] → Linear(64,2)
```

**Precision details:**

* Node counts are **fixed** (96, 116) ⇒ no `to\_dense\_batch` padding. `x.view(B, 96, 64)` after the PyG encoder. Verify the batch vector is contiguous.
* `nn.MultiheadAttention(embed\_dim=64, num\_heads=4, batch\_first=True)`. Q length ≠ KV length is fine; output length = Q length.
* **Norm: `BatchNorm`, not `GraphNorm`.** Already measured — GraphNorm suppressed pooled-vector SD 0.024 vs 0.199, and the swap moved LOSO AUC 0.473 → 0.551. Keep GraphNorm only as an ablation.
* Readout: mean-pool. Flatten was already tested and did not rescue it.

### 5.1 The §4.2 novelty is only testable if Graph 1 is genuinely per-subject

If Graph 1's topology were constant, `S` would be near-constant, and:

* Variant A = a **fixed query** attending over `F` → a fixed linear readout of `F`.
* Variant B = a varying query attending over **constant** values → a weighted average of a constant.

Either way the direction ablation compares a constant to a constant. **SubjectAdaptiveGraph is what makes §4.2 a real experiment.** The unit test in §4.1 is therefore load-bearing for the paper's headline claim, not a hygiene detail.

\---

## 6\. Protocol

* **Nested LOSO-site.** Outer: 17 folds, one site held out. Inner: **3-fold `StratifiedGroupKFold(groups=SITE\_ID)`** on the 16 training sites — no site appears in both inner-train and inner-val → select HP → refit → **freeze** → predict test site once. Plain stratified-by-subject was leak-prone (same site on both sides of the inner split, HP tuned on site-leaked validation). `StratifiedGroupKFold` groups by site *and* preserves class balance; site×DX is clean (p=0.267) so both constraints are satisfiable. 16 sites / 3 folds ≈ 5 held per inner fold.
* **Epoch selection: inner validation split ONLY.** Never the test site. This is the +0.129 leak.
* **Seeds:** 5 per outer fold, applied *after* HP selection.
* **ComBat / scaler / SubjectAdaptiveGraph projection / edge-ComBat: fit on train rows only, per fold.**

**Budget:**

```
per model: 17 outer × (8 HP configs × 3 inner) = 408 selection runs
         + 17 outer × 5 seeds                  =  85 final runs
                                               = 493 runs
5 GNN models (1a,1b,2,3,4) × 493 ≈ 2465 runs @ \~30 s ≈ 20.5 h
```

Plus **1a-MLP / 1b-MLP** parity controls: same nested loop, no message passing ⇒ \~seconds/run, negligible. Plus linear rungs (−2…0c): minutes.
Exceeds Kaggle's 12 h session ⇒ **checkpoint per outer fold to `/kaggle/working`, resume by fold index.** Cap `HP\_GRID` at 8 configs. Linear baselines are minutes.

\---

## 7\. Ablation ladder

Each rung isolates exactly one thing. **Every rung reports ROC-AUC *and* PR-AUC** (site/class imbalance makes ROC-AUC alone optimistic). **Rung −2 is the floor for every fMRI claim.**

|Rung|Model|Isolates|Expected|
|-|-|-|-|
|**−2**|LogReg on `\[func\_mean\_fd]`|**head motion alone**|**≈ 0.58**|
|**−1**|LogReg on `\[age, sex, FIQ, FD]`|phenotype (no site dummies — leaks under LOSO)|\~0.60|
|**0a**|svm\_rbf on 232-vector|sMRI linear|**0.6037** (known)|
|**0b**|SVM on 6,670 Fisher-z|**fMRI linear**|\~0.65–0.70 (lit.)|
|**0c**|SVM on `\[232 ‖ 6670]`|linear fusion|?|
|**1a**|sMRI GNN, `k` swept|structural graph bias vs 0a|?|
|**1a-MLP**|MLP, **same per-node encoder + mean-pool, no message passing**|**isolates edges** from the encoder on 1a|?|
|**1b**|fMRI GNN, `θ/k` swept|functional graph bias vs 0b|?|
|**1b-MLP**|MLP, same encoder + mean-pool, no message passing|**isolates edges** from the encoder on 1b|?|
|**2**|Dual-graph, concat (no attn)|fusion vs max(1a,1b)|?|
|**3**|+ cross-attn, Variant A|**novelty**|?|
|**4**|+ cross-attn, Variant B|CAS-GNN direction|?|

> \*\*Parity design — do not flatten.\*\* The MLP control must share 1a/1b's node encoder and mean-pool readout, ablating \*\*only\*\* the message-passing layers. Do \*\*not\*\* give it flattened `\[B, 96×64]` — that hands it 6,144 inputs vs the GNN's pooled 64, so it would win on capacity, not prove edges are useless. Same encoder, same pool, message-passing removed = the clean isolation of the graph's contribution.

**Claims and their tests:**

* **Graph bias earns its keep: `1b > 0b` AND `1b > 1b-MLP`** (both beyond MDE). The second conjunct is the real test — without it, `1b > 0b` could be the learned encoder, not the edges. Same for structural: `1a > 0a` AND `1a > 1a-MLP`.
* Fusion earns its keep: **2 vs max(1a, 1b)**
* Cross-attention earns its keep: **max(3,4) vs 2**
* Direction matters (§4.2 novelty): **3 vs 4**
* **Anything at all earns its keep: best vs Rung −2**

### 7.1 Mandatory ablations

* **A1** `k` sweep on Graph 1 `{5,10,20,30}`
* **A2** `θ` / `k\_func` sweep on Graph 2
* **A3** full-T vs truncated-T\* (precision cost of the §2/§3.2 CDF-percentile truncation)
* **A4** edge-ComBat on/off (§3.2 F5)
* **A5** GraphNorm vs BatchNorm (confirm the 0.473→0.551 finding holds)

\---

## 8\. Statistics

* **Unit:** site-level AUC (ROC **and** PR), averaged over 5 seeds → **n = 17 paired observations.**
* **Test:** Wilcoxon signed-rank, paired by site.
* **Metrics:** ROC-AUC (primary) + **PR-AUC (secondary, pre-registered)** — reported for every rung, both entered into the paired tests.
* **Correction:** Holm–Bonferroni across the 5 claims in §7.
* **Effect size:** median ΔAUC + 95% BCa bootstrap CI over sites.

**Minimum detectable effect — pre-register this. The prior 0.02–0.03 figure was wrong (over-optimistic).**
Observed across-site AUC SD in the structural results ≈ **0.087**. At n=17, α=0.05 two-sided, Holm-corrected, power 0.80 ⇒ realistic **MDE ≈ 0.06–0.07 AUC** — roughly 3× the earlier claim.

> \*\*One caveat, stated so it isn't misread as a fixed number.\*\* The paired test's power depends on `SD(ΔAUC)`, the SD of the \*per-site difference\*, not on raw per-site AUC SD. Because compared models share sites/subjects they will be positively correlated, so `SD(ΔAUC) = √(σ²\_A + σ²\_B − 2·cov)` is typically \*\*smaller\*\* than raw SD — but if the models were independent it rises to `√2 × 0.087 ≈ 0.123` (MDE ≈ 0.10). So \*\*0.06–0.07 is the planning estimate; recompute the true MDE post-hoc from the observed ΔAUC SD per comparison.\*\* Bounds: \~0.04 (highly correlated) to \~0.10 (independent).

**You cannot detect a true effect below \~0.06 AUC.** CAS-GNN's cross-attention change was **\~0.03 AUC** — **below your resolution**. Any null at that scale is **"no evidence for an effect," not "evidence of no effect."** Pre-register this sentence or the null is uninterpretable.

\---

## 9\. Motion \& physiological-plausibility gates

FD is confounded at **r = +0.138, p = 0.0001**. These are pass/fail gates, not appendix material.

### 9.1 The motion floor

```
r\_pb = 0.138
d    = 2r/√(1−r²) = 0.276/0.9904 = 0.2787
AUC  = Φ(d/√2)    = Φ(0.1971)    ≈ 0.578
```

**Head motion alone ≈ 0.58 AUC.** A model at 0.65 is claiming **+0.07 over fidgeting**. Report it that way or a reviewer will. **This floor is N-independent** — derived from the measured `r=+0.138`, not from cohort size — so it stands regardless of Stage 2's Nf. It is the one hard number the plan can already assert.

### 9.2 G1 — Score–motion correlation (sharpest test)

On held-out sites, compute `r(ŷ\_prob, func\_mean\_fd)`. **If this is significant, your classifier is partly a motion detector.** Report per site and pooled.

### 9.3 G2 — Motion-matched subsample

Propensity-match ASD/TD on FD **within site**. Re-run the full LOSO on the matched subset. Report ΔAUC vs unmatched. Expect a drop; the size of the drop *is* the motion contribution.

### 9.4 G3 — FD as protected covariate

Include `func\_mean\_fd` in ComBat's protected set (§3.2 F5) **and** as a node-level feature on Graph 2. Report with/without.

### 9.5 G4 — Power motion signature (the plausibility check)

Motion inflates short-range FC and reduces long-range FC (Power 2012; Van Dijk 2012). Using AAL centroids:

```
fc\_short = mean(fc\_z\[d\_ij <  30 mm])
fc\_long  = mean(fc\_z\[d\_ij >  90 mm])
```

Correlate each with FD. Then extract the model's top-attended edges (attention weights from §5) and test whether they **align with the motion signature** rather than with known ASD networks (default-mode, salience). **If the model's important edges are the motion signature, the result is an artifact regardless of AUC.**

### 9.6 G5 — Deployment metrics

Report for every GNN rung: **parameter count, FLOPs, inference latency (ms, batch=1, CPU and GPU).** A dual-graph cross-attention model that needs 500 ms/subject is not a BCI-adjacent contribution.

\---

## 10\. Stage-by-stage execution

|Stage|Action|Gate to pass|Cost|
|-|-|-|-|
|**0**|**τ sweep on existing HeteroGNN** `{0.3,0.5,0.6,0.7,0.8}` × 17 folds × 3 seeds|Does AUC move off 0.5557?|\~3 h|
|**1**|Answer B1–B3 (§1)|`label` is not `DX\_GROUP`|—|
|**2**|Freeze cohort → `cohort\_final.csv` (SUB\_ID, FILE\_ID, SITE\_ID, DX, age, sex, FIQ, FD, T)|Σ=Nf, printed by Stage 2|1 h|
|**3**|F1–F4 → `fc\_z.npy` `\[Nf,116,116]`|dead-ROI count not site-systematic|2 h|
|**4**|Rungs −2, −1, 0a, 0b, 0c|**0b > −2 by ≥ MDE**, else fMRI adds nothing over motion|2 h|
|**5**|`SubjectAdaptiveGraph` + **unit test: `std(edge\_index) > 0` across subjects**|test passes|3 h|
|**6**|Rungs 1a, 1b + sweeps A1, A2|**1b > 0b?** — if no, the graph premise is dead|8 h|
|**7**|Rung 2 (fusion, no attention)|2 > max(1a,1b)?|4 h|
|**8**|Rungs 3, 4 (§4.2 novelty)|3 vs 4|8 h|
|**9**|Gates G1–G5|**G1 non-significant, G4 not motion-aligned**|6 h|
|**10**|A3, A4, A5|—|6 h|

**Stage 0 gates the paper's narrative, not the architecture.** You are replacing group-level graphs with `SubjectAdaptiveGraph`, so τ is moot for the *new* design — but the HeteroGNN null is your Section-2 motivation, and it is currently unearned. Three hours converts an assertion into a finding.

**Stage 4 is the hard gate.** If `0b − (−2) < MDE`, then functional connectivity adds nothing over head motion in this cohort, and no architecture rescues that.

**Stage 6 is the thesis gate.** If `1b ≤ 0b`, the graph inductive bias does not help at N=780 on *either* modality, tested properly this time. That is a genuine, publishable null — and unlike the current one, it would be earned.

\---

## 11\. Kill criteria — decide now, not after seeing results

|If|Then|
|-|-|
|Stage 4: `0b ≤ −2 + MDE`|fMRI adds nothing over motion. **Stop. Write the negative result.**|
|Stage 6: `1b ≤ 0b` and `1a ≤ 0a`|Graph bias fails on both modalities, properly tested. **Publish the null** — it is stronger than the current unearned one.|
|Stage 8: `\|3 − 4\| < MDE`|Direction does not matter. Report as **underpowered**, not as absence (§8).|
|Gate G1 significant|Model reads motion. **No AUC claim is reportable** until G3 clears.|

\---

## 12\. Deliverables

* `cohort\_final.csv`, `fc\_z.npy` (+ SHA256)
* `results/ladder.csv` — `model × site × seed × auc × params × flops × latency\_ms`
* `results/motion\_gates.csv` — G1–G4
* `results/ablations.csv` — A1–A5
* `requirements.txt`, seed list, single reproducing command
* Pre-registered MDE (§8) written **before** Stage 4 runs

\---

## Appendix — what this plan will and will not let you claim

**Can claim:** a leakage-free 17-site nested-LOSO benchmark; a quantified protocol finding (epoch selection on the test fold inflates multi-site ASD classification by **+0.129 AUC, p=0.031**, enough to manufacture an architecture's worth of apparent improvement); a motion-controlled multimodal comparison; whether graph inductive bias earns its parameters at N≈780.

**Cannot claim:** that the graph "adds information" (§4.3); that results generalise beyond **low-motion ASD** (§2.2); a null on any effect below **\~0.06 AUC** (§8) — including CAS-GNN-scale cross-attention (\~0.03), which is below resolution.

