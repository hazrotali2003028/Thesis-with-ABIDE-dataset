"""
ComBat harmonization with train-only fit and forward application to an
UNSEEN batch (plan Stage 4, under the Stage 5 LOSO constraint).

Why this exists instead of neuroCombat
--------------------------------------
neuroCombat.neuroCombatFromTraining() raises

    ValueError: The batches ['X'] are not part of the training dataset

so it cannot harmonize a site that was held out — which is exactly what
leave-one-site-out does on every fold. (The function also announces itself as
"[neuroCombatFromTraining] In development ...".)

The resolution, per the project decision:
  - The standardization (B_hat, grand_mean, var_pooled) and the empirical-Bayes
    hyper-priors are estimated on TRAINING sites only, with DX + age + sex in
    the design matrix so diagnosis variance is protected during batch-parameter
    estimation.
  - The held-out site's own batch parameters (gamma, delta) are estimated from
    ITS OWN FEATURES. No label is used, so no label leakage.

Two consequences a reviewer will ask about, stated plainly:
  1. This is TRANSDUCTIVE. Estimating a site's mean/variance needs the site's
     subjects as a group, so a single walk-in subject cannot be harmonized.
     Any clinical-deployment framing must be dropped.
  2. The test site's standardization omits the DX term (no labels available),
     while training's includes it. The diagnosis effect is therefore preserved
     exactly on train and scaled by 1/sqrt(delta*) on test. Since delta* ~= 1
     the distortion is small, but it is an approximation, not an identity.

Reference: Johnson, Li & Rabinovic (2007), Biostatistics 8(1):118-127.
"""

import numpy as np


# ─── EMPIRICAL BAYES ─────────────────────────────────────────────────────────
def _a_prior(delta_hat):
    # ddof=1 matches the reference implementation; ddof=0 shifts the inverse
    # gamma prior enough to move harmonized values in the 3rd decimal.
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (2 * s2 + m ** 2) / s2


def _b_prior(delta_hat):
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (m * s2 + m ** 3) / s2


def _convert_zeroes(d):
    """A feature that is constant within a batch gives delta_hat = 0, which
    would divide by zero downstream. The reference implementation maps it to 1."""
    d = d.copy()
    d[d == 0] = 1.0
    return d


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(z, g_hat, d_hat, g_bar, t2, a, b, tol=1e-4, max_iter=500):
    """Iterative EB solution for one batch. z is (n_samples, n_features).

    g_bar, t2, a, b are SCALARS: ComBat's empirical Bayes pools across the
    features within a batch, which is where the shrinkage strength comes from.
    Pooling across batches per-feature instead is a different (wrong) estimator.
    """
    n = z.shape[0]
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((z - g_new) ** 2).sum(axis=0)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
            np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)),
        )
        g_old, d_old = g_new, d_new
        if change < tol:
            break
    return g_old, d_old


# ─── PUBLIC API ──────────────────────────────────────────────────────────────
def combat_fit(Y, batch, X_protect, X_nolabel_idx):
    """Fit ComBat on training data only.

    Args:
        Y:             (n, p) float array of features.
        batch:         (n,) array of site labels.
        X_protect:     (n, k) design of protected covariates, INCLUDING the
                       diagnosis column. Must NOT contain an intercept.
        X_nolabel_idx: indices of columns in X_protect that are available at
                       test time (i.e. everything except diagnosis).

    Returns:
        dict of estimates for combat_apply_unseen / combat_apply_train.
    """
    Y = np.asarray(Y, dtype=np.float64)
    sites = np.unique(batch)
    n, p = Y.shape

    # Design: full one-hot over batches (no reference level) + protected covars
    D_batch = np.zeros((n, len(sites)))
    for j, s in enumerate(sites):
        D_batch[batch == s, j] = 1.0
    design = np.hstack([D_batch, X_protect])

    # Least squares
    B_hat = np.linalg.lstsq(design, Y, rcond=None)[0]

    # Grand mean weighted by batch size; var_pooled from full-design residuals
    n_per = D_batch.sum(axis=0)
    grand_mean = (n_per / n) @ B_hat[: len(sites)]
    resid = Y - design @ B_hat
    var_pooled = (resid ** 2).sum(axis=0) / n

    B_cov = B_hat[len(sites):]                       # covariate coefficients
    stand_mean = grand_mean + X_protect @ B_cov      # train uses full design
    Z = (Y - stand_mean) / np.sqrt(var_pooled)

    # Per-batch L/S parameters
    gamma_hat = np.vstack([Z[batch == s].mean(axis=0) for s in sites])
    delta_hat = np.vstack([_convert_zeroes(Z[batch == s].var(axis=0, ddof=1))
                           for s in sites])

    # EB hyper-priors are per-batch, pooled ACROSS FEATURES
    gamma_star, delta_star = {}, {}
    for j, s in enumerate(sites):
        gs, ds = _it_sol(
            Z[batch == s], gamma_hat[j], delta_hat[j],
            gamma_hat[j].mean(), gamma_hat[j].var(ddof=1),
            _a_prior(delta_hat[j]), _b_prior(delta_hat[j]),
        )
        gamma_star[s] = gs
        delta_star[s] = ds

    return {
        "sites": sites,
        "B_cov": B_cov,
        "grand_mean": grand_mean,
        "var_pooled": var_pooled,
        "gamma_star": gamma_star,
        "delta_star": delta_star,
        "X_nolabel_idx": np.asarray(X_nolabel_idx),
    }


def combat_apply_train(Y, batch, X_protect, est):
    """Apply the fitted model back onto the training sites."""
    Y = np.asarray(Y, dtype=np.float64)
    stand_mean = est["grand_mean"] + X_protect @ est["B_cov"]
    Z = (Y - stand_mean) / np.sqrt(est["var_pooled"])
    out = np.empty_like(Z)
    for s in est["sites"]:
        m = batch == s
        out[m] = (Z[m] - est["gamma_star"][s]) / np.sqrt(est["delta_star"][s])
    return out * np.sqrt(est["var_pooled"]) + stand_mean


def combat_apply_unseen(Y, X_protect, est):
    """Harmonize a site that was NOT in the training fit.

    The site's gamma/delta are estimated from its own features. Labels are
    never touched: the DX column of X_protect is masked out via
    est['X_nolabel_idx'].
    """
    Y = np.asarray(Y, dtype=np.float64)
    if len(Y) < 2:
        raise ValueError(
            "combat_apply_unseen is transductive and needs >=2 subjects from "
            f"the held-out site to estimate its batch parameters; got {len(Y)}."
        )

    # Standardize using training coefficients, label-free covariates only
    idx = est["X_nolabel_idx"]
    B_cov_nolabel = est["B_cov"][idx]
    stand_mean = est["grand_mean"] + X_protect[:, idx] @ B_cov_nolabel
    Z = (Y - stand_mean) / np.sqrt(est["var_pooled"])

    # Batch parameters from this site's own data. The EB priors are per-batch
    # quantities pooled across features, so the held-out site supplies its own
    # — still label-free, and identical in form to what each training site got.
    gamma_hat = Z.mean(axis=0)
    delta_hat = _convert_zeroes(Z.var(axis=0, ddof=1))
    gamma_star, delta_star = _it_sol(
        Z, gamma_hat, delta_hat,
        gamma_hat.mean(), gamma_hat.var(ddof=1),
        _a_prior(delta_hat), _b_prior(delta_hat),
    )

    Z_adj = (Z - gamma_star) / np.sqrt(delta_star)
    return Z_adj * np.sqrt(est["var_pooled"]) + stand_mean
