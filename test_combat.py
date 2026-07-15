import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\ABIDE DATASET\Thesis")
from combat import combat_fit, combat_apply_train, combat_apply_unseen
from neuroCombat import neuroCombat

np.random.seed(1)
n, p = 300, 12
site = np.random.choice(["A", "B", "C", "D"], n)
age = np.random.uniform(6, 40, n)
sex = np.random.binomial(1, 0.5, n)
dx = np.random.binomial(1, 0.5, n)
shift = {"A": 0.0, "B": 3.0, "C": -2.0, "D": 5.0}
scale = {"A": 1.0, "B": 2.0, "C": 0.5, "D": 1.5}
X = (np.random.randn(n, p) * np.array([scale[s] for s in site])[:, None]
     + np.array([shift[s] for s in site])[:, None])
X += 0.4 * dx[:, None] + 0.02 * age[:, None]

Xprot = np.column_stack([dx, age, sex]).astype(float)
NOLABEL = [1, 2]   # age, sex (dx is column 0)

# ─── 1. Does my ComBat match neuroCombat on the SAME data? ───────────────────
print("=" * 66)
print("TEST 1: agreement with neuroCombat (all 4 sites, same design)")
print("=" * 66)
ref = neuroCombat(
    dat=X.T,
    covars=pd.DataFrame({"batch": site, "DX": dx, "AGE": age, "SEX": sex}),
    batch_col="batch",
    categorical_cols=["DX", "SEX"],
    continuous_cols=["AGE"],
)["data"].T

est = combat_fit(X, site, Xprot, NOLABEL)
mine = combat_apply_train(X, site, Xprot, est)

diff = np.abs(ref - mine)
print(f"  max abs diff vs neuroCombat : {diff.max():.3e}")
print(f"  mean abs diff               : {diff.mean():.3e}")
print(f"  correlation                 : {np.corrcoef(ref.ravel(), mine.ravel())[0,1]:.8f}")
print("  VERDICT:", "MATCH" if diff.max() < 1e-6 else "MISMATCH — investigate")

# ─── 2. Train-only fit + forward apply to the UNSEEN site D ──────────────────
print("\n" + "=" * 66)
print("TEST 2: fit on A,B,C -> harmonize UNSEEN site D (neuroCombat cannot)")
print("=" * 66)
tr, te = site != "D", site == "D"
est2 = combat_fit(X[tr], site[tr], Xprot[tr], NOLABEL)
Htr = combat_apply_train(X[tr], site[tr], Xprot[tr], est2)
Hte = combat_apply_unseen(X[te], Xprot[te], est2)

def cv(a):
    return np.std(a) / (np.abs(np.mean(a)) + 1e-9)

print("  feature 0 site means BEFORE:",
      {s: round(X[site == s, 0].mean(), 2) for s in "ABCD"})
after = {s: round(Htr[site[tr] == s, 0].mean(), 2) for s in "ABC"}
after["D"] = round(Hte[:, 0].mean(), 2)
print("  feature 0 site means AFTER :", after)

print("\n  between-site variance of site means (mean over features):")
before_v = np.mean([np.var([X[site == s, j].mean() for s in "ABCD"]) for j in range(p)])
allh = np.zeros_like(X); allh[tr] = Htr; allh[te] = Hte
after_v = np.mean([np.var([allh[site == s, j].mean() for s in "ABCD"]) for j in range(p)])
print(f"    before = {before_v:8.4f}")
print(f"    after  = {after_v:8.4f}   ({100*(1-after_v/before_v):.1f}% reduction)")

# ─── 3. Leakage guard: does the test-site output depend on its labels? ───────
print("\n" + "=" * 66)
print("TEST 3: leakage guard — permute test-site labels, output must not move")
print("=" * 66)
Xprot_perm = Xprot.copy()
rng = np.random.default_rng(7)
Xprot_perm[te, 0] = rng.permutation(Xprot_perm[te, 0])   # scramble DX on site D
Hte_perm = combat_apply_unseen(X[te], Xprot_perm[te], est2)
d = np.abs(Hte - Hte_perm).max()
print(f"  max abs change after permuting test DX: {d:.3e}")
print("  VERDICT:", "NO LEAKAGE" if d == 0.0 else "LEAK — test labels affect output")

# ─── 4. Transductive guard ───────────────────────────────────────────────────
print("\n" + "=" * 66)
print("TEST 4: single-subject harmonization must be refused")
print("=" * 66)
try:
    combat_apply_unseen(X[te][:1], Xprot[te][:1], est2)
    print("  VERDICT: FAIL — accepted a single subject")
except ValueError as e:
    print("  raised ValueError as intended:")
    print("   ", str(e)[:90])
