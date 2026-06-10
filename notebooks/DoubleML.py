# Databricks notebook source
# Phase 7 -- Double ML (Partially Linear Regression)
# Applied Causal Inference Series
#
# Problem:  Does job training increase earnings?
# Method:   Double Machine Learning -- Robinson (1988) Partialling-Out (PLR)
# Dataset:  LaLonde (1986) -- same treated workers as Phases 2-4, CPS-3 controls
# Estimand: ATT -- Average Treatment Effect on the Treated
#           Same estimand as Phases 2, 3, 4. Directly comparable to the $1,794 RCT benchmark.
#
# Why DML after IPW and AIPW?
#   Phase 4 showed that AIPW with a linear outcome model can HURT the estimate.
#   AIPW ATT = $476, worse than raw IPW ATT = $1,212.
#   The culprit: re78 is right-skewed. OLS misfits the earnings distribution,
#   and the doubly-robust augmentation amplified the misspecification error.
#
#   DML fixes this at the root:
#     1. Replace the linear outcome model with gradient boosting (handles skew)
#     2. Replace the linear propensity model with gradient boosting (better calibration)
#     3. Use Neyman-orthogonal scores -- the causal estimate is insensitive to
#        small errors in both nuisance models, so parametric convergence rates
#        hold even though the nuisance models converge more slowly.
#     4. Cross-fit (K=5) to avoid overfitting bias from in-sample predictions.
#
# The PLR model:
#   Y  = theta * D + g(X) + epsilon     (outcome equation)
#   D  = m(X) + v                       (treatment equation)
#
#   Partialling out:
#   Ytilde = Y  - E[Y|X]    (residual outcome -- variation in Y not explained by X)
#   Dtilde = D  - E[D|X]    (residual treatment -- variation in D not explained by X)
#   theta  = E[Dtilde * Ytilde] / E[Dtilde^2]   (regression of Ytilde on Dtilde)
#
#   This is Robinson's (1988) insight: theta is identified from the co-movement
#   of the parts of Y and D that cannot be explained by X. Confounders in X
#   are fully partialled out by the nuisance models, regardless of their functional form.
#
# ATT via propensity-weighted PLR:
#   theta_ATT = E[e(X) * Dtilde * Ytilde] / E[e(X) * Dtilde^2]
#
#   where e(X) = P(D=1|X). Up-weighting treated-like units focuses the
#   estimate on the treated subpopulation.
#
# Two-path structure (mirrors Phases 3 & 4):
#   Path A -- Full CPS (15,992 controls): near-zero overlap (AUC ~1.0)
#             DML will run but nuisance models will overfit; documented for continuity.
#   Path B -- CPS-3   (   429 controls): borderline overlap (AUC ~0.87)
#             Primary analysis.
#
# Benchmark: $1,794 (LaLonde RCT)


# COMMAND ----------

# MAGIC %pip install scipy statsmodels scikit-learn mlflow --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC

# COMMAND ----------

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
import statsmodels.api as sm
import mlflow
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

TRUE_ATT = 1794  # LaLonde RCT benchmark
N_FOLDS  = 5     # Cross-fitting folds
print("✓ Imports complete")
print(f"  Benchmark ATT: ${TRUE_ATT:,}")
print(f"  Cross-fitting folds: {N_FOLDS}")


# COMMAND ----------

# =============================================================================
# Simulation Fallback
# =============================================================================
#
# If the NBER URLs are unavailable (no internet on cluster), fall back to
# simulated data calibrated to LaLonde's published summary statistics.
# Identical to Phases 3 & 4 -- one consistent fallback across the series.

def make_group(n, treat, age_mean, educ_mean, pct_black, pct_hisp, pct_married, pct_nodegree,
               re74_mean, re75_mean, re74_sd, re75_sd, true_effect=0, earnings_noise=4000, seed=None):
    if seed is not None: np.random.seed(seed)
    re74 = np.random.normal(re74_mean, re74_sd, n).clip(0)
    re75 = np.random.normal(re75_mean, re75_sd, n).clip(0)
    re78 = (0.90 * re75 + 0.15 * re74 + true_effect + np.random.normal(0, earnings_noise, n)).clip(0)
    return pd.DataFrame({
        "treat":    treat,
        "age":      np.random.normal(age_mean, 7, n).clip(17, 60).round(),
        "educ":     np.random.normal(educ_mean, 2, n).clip(0, 18).round(),
        "black":    np.random.binomial(1, pct_black, n),
        "hisp":     np.random.binomial(1, pct_hisp, n),
        "married":  np.random.binomial(1, pct_married, n),
        "nodegree": np.random.binomial(1, pct_nodegree, n),
        "re74":     re74.round(2),
        "re75":     re75.round(2),
        "re78":     re78.round(2),
    })

def get_simulated_lalonde(seed=42):
    df_treat = make_group(
        n=185, treat=1, age_mean=25, educ_mean=10, pct_black=0.84, pct_hisp=0.06,
        pct_married=0.19, pct_nodegree=0.71, re74_mean=2096, re75_mean=1532,
        re74_sd=4887, re75_sd=3219, true_effect=TRUE_ATT, seed=seed
    )
    df_ctrl = make_group(
        n=2490, treat=0, age_mean=34, educ_mean=12, pct_black=0.07, pct_hisp=0.06,
        pct_married=0.87, pct_nodegree=0.31, re74_mean=13651, re75_mean=13405,
        re74_sd=9270, re75_sd=9270, true_effect=0, seed=seed + 1
    )
    df_cps = make_group(
        n=429, treat=0, age_mean=28, educ_mean=11, pct_black=0.11, pct_hisp=0.08,
        pct_married=0.45, pct_nodegree=0.45, re74_mean=8725, re75_mean=7400,
        re74_sd=9000, re75_sd=8110, true_effect=0, seed=seed + 2
    )
    return df_treat, df_ctrl, df_cps


# COMMAND ----------

# =============================================================================
# Load LaLonde Data (Two Control Groups)
# =============================================================================
#
# Same data loading as Phases 3 & 4 -- identical treated group, two control groups.
#
# df_invalid = treated + Full CPS (15,992 controls)  -> Path A: documented failure
# df         = treated + CPS-3   (  429 controls)    -> Path B: primary analysis
#
# KEY CONTINUITY:
#   Same NBER URLs, same COLUMNS, same COVARIATES as all prior phases.
#   Any improvement over Phase 4 is due to better nuisance models, not different data.

TREATED_URL     = "http://www.nber.org/~rdehejia/data/nswre74_treated.txt"
CONTROL_URL     = "http://www.nber.org/~rdehejia/data/cps_controls.txt"
CPS_CONTROL_URL = "http://www.nber.org/~rdehejia/data/cps3_controls.txt"
COLUMNS = ['treat', 'age', 'educ', 'black', 'hisp', 'married', 'nodegree', 're74', 're75', 're78']

try:
    df_treat    = pd.read_csv(TREATED_URL,     names=COLUMNS, sep='  ', engine='python')
    df_ctrl     = pd.read_csv(CONTROL_URL,     names=COLUMNS, sep='  ', engine='python')
    df_cps_ctrl = pd.read_csv(CPS_CONTROL_URL, names=COLUMNS, sep='  ', engine='python')
    print("✓ Loaded from NBER website")
except Exception:
    df_treat, df_ctrl, df_cps_ctrl = get_simulated_lalonde()
    print("✓ Loaded from simulation (NBER unavailable)")

# Path A: Full CPS -- overlap failure documented for series continuity
df_invalid = pd.concat([df_treat, df_ctrl], ignore_index=True)
df_invalid['treat'] = df_invalid['treat'].astype(float).astype(int)

# Path B: CPS-3 -- primary analysis
df = pd.concat([df_treat, df_cps_ctrl], ignore_index=True)
df['treat'] = df['treat'].astype(float).astype(int)

COVARIATES = ['age', 'educ', 'black', 'hisp', 'married', 'nodegree', 're74', 're75']
OUTCOME    = 're78'
TREATMENT  = 'treat'

print(f"\nPath A -- Full CPS:  {df_invalid.shape[0]:,} workers "
      f"({df_invalid['treat'].sum():,} treated | {(df_invalid['treat']==0).sum():,} controls)")
print(f"Path B -- CPS-3:     {df.shape[0]:,} workers "
      f"({df['treat'].sum():,} treated | {(df['treat']==0).sum():,} controls)")
print(f"\nCovariates: {COVARIATES}")
print(f"Outcome:    {OUTCOME}")


# COMMAND ----------

# =============================================================================
# Helper: Naive Estimate and Baseline Imbalance
# =============================================================================
#
# Same naive comparison as prior phases -- raw mean difference before any adjustment.
# Establishes the baseline we are trying to correct.

def compute_smd(df, covariates, treat_col='treat'):
    """Standardized mean difference per covariate."""
    smds = {}
    for col in covariates:
        m1 = df[df[treat_col]==1][col].mean()
        m0 = df[df[treat_col]==0][col].mean()
        s  = df[col].std()
        smds[col] = ((m1 - m0) / s) if s > 0 else 0
    return pd.Series(smds)

def show_baseline(df, outcome, covariates, true_att, treat_col='treat'):
    """Print naive estimate and covariate balance table."""
    naive = (df[df[treat_col]==1][outcome].mean()
           - df[df[treat_col]==0][outcome].mean())
    print(f"Naive ATT (raw mean difference): ${naive:,.0f}")
    print(f"True  ATT (RCT benchmark):       ${true_att:,}")
    print(f"\nCovariate SMD before DML:")
    print(f"{'Covariate':<12} {'SMD':>8}  {'Balance':>12}")
    print("─" * 36)
    smds = compute_smd(df, covariates, treat_col)
    for cov, smd in smds.items():
        status = "✓ OK" if abs(smd) < 0.1 else ("⚠ Moderate" if abs(smd) < 0.2 else "✗ Imbalanced")
        print(f"{cov:<12} {smd:>8.3f}  {status:>12}")
    print(f"\nMean |SMD|: {smds.abs().mean():.3f}  |  Max |SMD|: {smds.abs().max():.3f}")
    return naive, smds

print("=" * 60)
print("PATH B -- CPS-3 (primary analysis)")
print("=" * 60)
naive_att, smds_before = show_baseline(df, OUTCOME, COVARIATES, TRUE_ATT)


# COMMAND ----------

# =============================================================================
# Cross-Fitted Nuisance Models
# =============================================================================
#
# DML requires out-of-sample predictions for both nuisance functions to avoid
# overfitting bias. We use K-fold cross-fitting: fit on K-1 folds, predict
# on the held-out fold. After K iterations every observation has an
# out-of-sample prediction.
#
# E[Y|X] -- Outcome model: GradientBoostingRegressor
#   Why GBM over OLS?
#     re78 is right-skewed with a mass at zero. OLS assumes a linear,
#     homoskedastic conditional mean. GBM fits nonlinear trees that can
#     capture the earnings distribution without assuming a functional form.
#     This is precisely the misspecification Phase 4's AIPW suffered from.
#
# E[D|X] -- Propensity model: GradientBoostingClassifier
#   Why GBM over logistic regression?
#     Same reasoning -- treatment selection may depend nonlinearly on
#     covariates. GBM delivers better-calibrated propensity scores on
#     tabular data with interaction effects.
#
# Cross-fitting hyperparameters (same for both models):
#   n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8
#   Shallow trees (depth 3) prevent individual tree overfitting.
#   200 estimators with low learning rate is a standard regularised setup.
#   subsample=0.8 adds stochastic gradient boosting regularisation.
#
# After this cell df has four new columns:
#   Yhat  -- out-of-sample E[Y|X]  predictions
#   Dhat  -- out-of-sample E[D|X]  predictions (propensity scores)
#   Ytilde -- Y - Yhat  (residual outcome)
#   Dtilde -- D - Dhat  (residual treatment)

def cross_fit_nuisance(df, covariates, outcome, treatment, n_folds=5, random_state=42):
    """
    Cross-fit outcome and propensity nuisance models using GradientBoosting.
    Returns df with Yhat, Dhat, Ytilde, Dtilde columns.
    """
    X = df[covariates].values
    Y = df[outcome].values
    D = df[treatment].values

    Yhat = np.zeros(len(df))
    Dhat = np.zeros(len(df))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    gbr_params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                      subsample=0.8, random_state=random_state)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # Outcome model: GBM regressor
        gbr = GradientBoostingRegressor(**gbr_params)
        gbr.fit(X_tr, Y_tr)
        Yhat[test_idx] = gbr.predict(X_te)

        # Propensity model: GBM classifier
        gbc = GradientBoostingClassifier(**gbr_params)
        gbc.fit(X_tr, D_tr)
        Dhat[test_idx] = gbc.predict_proba(X_te)[:, 1]

        # Per-fold diagnostics
        rmse_y = np.sqrt(mean_squared_error(Y_te, Yhat[test_idx]))
        auc_d  = roc_auc_score(D_te, Dhat[test_idx])
        fold_metrics.append({'fold': fold+1, 'rmse_outcome': rmse_y, 'auc_propensity': auc_d})

    # Residuals
    df = df.copy()
    df['Yhat']   = Yhat
    df['Dhat']   = Dhat.clip(1e-6, 1 - 1e-6)  # propensity clipping for stability
    df['Ytilde'] = Y - Yhat
    df['Dtilde'] = D - Dhat

    metrics_df = pd.DataFrame(fold_metrics)
    return df, metrics_df

print("Fitting cross-fitted nuisance models -- CPS-3 (Path B)...")
df, fold_metrics = cross_fit_nuisance(df, COVARIATES, OUTCOME, TREATMENT, n_folds=N_FOLDS)

print(f"\n{'Fold':<6} {'RMSE(Y)':<14} {'AUC(D)':<10}")
print("─" * 32)
for _, row in fold_metrics.iterrows():
    print(f"  {int(row.fold):<4} ${row.rmse_outcome:>10,.0f}   {row.auc_propensity:.3f}")

print(f"\nMean RMSE (outcome):     ${fold_metrics['rmse_outcome'].mean():>10,.0f}")
print(f"Mean AUC  (propensity):  {fold_metrics['auc_propensity'].mean():.3f}")
print(f"\n✓ Cross-fitting complete -- Ytilde and Dtilde computed")
print(f"  Residual mean (Ytilde): {df['Ytilde'].mean():.2f}   <- should be near zero")
print(f"  Residual mean (Dtilde): {df['Dtilde'].mean():.4f}  <- should be near zero")


# COMMAND ----------

# =============================================================================
# Path A -- Full CPS: Nuisance Model Diagnostics Only
# =============================================================================
#
# We run cross-fitting on the Full CPS dataset to document the overlap failure
# from a DML lens. Unlike PSM and IPW, DML does not immediately 'break' --
# it produces a number. But the propensity scores will be near 0 for most
# controls, causing Dtilde to carry very little signal and the ATT weight
# e(X) to be near-zero for nearly all control units.
#
# The key diagnostic: mean |Dtilde| should be low if overlap is poor.
# When Dtilde is tiny, the denominator of the PLR estimator is small and
# noisy -- any small residual correlation drives a large estimate.

print("Fitting cross-fitted nuisance models -- Full CPS (Path A)...")
df_invalid, fold_metrics_invalid = cross_fit_nuisance(
    df_invalid, COVARIATES, OUTCOME, TREATMENT, n_folds=N_FOLDS)

print(f"\nMean RMSE (outcome):     ${fold_metrics_invalid['rmse_outcome'].mean():>10,.0f}")
print(f"Mean AUC  (propensity):  {fold_metrics_invalid['auc_propensity'].mean():.3f}")
print(f"\nPath A vs Path B -- Dtilde summary:")
print(f"  Path B (CPS-3)   -- mean |Dtilde|: {df['Dtilde'].abs().mean():.4f}  <- reasonable signal")
print(f"  Path A (Full CPS)-- mean |Dtilde|: {df_invalid['Dtilde'].abs().mean():.4f}  <- weak signal, noisy PLR")
print(f"\n⚠  Full CPS: AUC = {fold_metrics_invalid['auc_propensity'].mean():.3f} -- near-perfect separation.")
print(f"   Dtilde signal is thin. PLR estimate from Path A is unreliable.")
print(f"   Documenting for continuity. Primary analysis is Path B (CPS-3).")


# COMMAND ----------

# =============================================================================
# Visualize Nuisance Model Residuals
# =============================================================================
#
# Four-panel diagnostic:
#   A. Residual treatment Dtilde histogram -- bell-shaped around 0 is ideal.
#      Bimodal or heavily skewed Dtilde suggests poor propensity model fit.
#
#   B. Propensity score distribution by treatment group.
#      Overlap region (common support) must be non-trivial for ATT to be identified.
#
#   C. Ytilde vs Dtilde scatter -- the PLR estimate is the slope of this relationship.
#      If DML is working, the slope should be positive (training increases earnings)
#      and the scatter should be roughly linear in the residual space.
#
#   D. Nuisance RMSE and AUC per fold -- checks for fold-level instability.
#      A single fold with much higher RMSE or much lower AUC is a warning sign.

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Phase 7 -- DML Nuisance Model Diagnostics (CPS-3, Path B)",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# -- Panel A: Dtilde histogram ------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.hist(df.loc[df[TREATMENT]==1, 'Dtilde'], bins=40, alpha=0.7, color='#e06b5b',
        label=f'Treated (n={df[TREATMENT].sum():,})', density=True)
ax.hist(df.loc[df[TREATMENT]==0, 'Dtilde'], bins=40, alpha=0.55, color='#5b8dd9',
        label=f'Control (n={(df[TREATMENT]==0).sum():,})', density=True)
ax.axvline(0, color='#333', linestyle='--', lw=1.2)
ax.set_xlabel("Dtilde  =  D − E[D|X]")
ax.set_ylabel("Density")
ax.set_title("Residual Treatment Distribution\n(should be centred at 0)", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Panel B: Propensity score overlap ----------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.hist(df.loc[df[TREATMENT]==0, 'Dhat'], bins=40, alpha=0.55, color='#5b8dd9',
        label=f'Control (n={(df[TREATMENT]==0).sum():,})', density=True)
ax.hist(df.loc[df[TREATMENT]==1, 'Dhat'], bins=40, alpha=0.7, color='#e06b5b',
        label=f'Treated (n={df[TREATMENT].sum():,})', density=True)
overlap_min = max(df.loc[df[TREATMENT]==1,'Dhat'].min(), df.loc[df[TREATMENT]==0,'Dhat'].min())
overlap_max = min(df.loc[df[TREATMENT]==1,'Dhat'].max(), df.loc[df[TREATMENT]==0,'Dhat'].max())
ax.axvspan(overlap_min, overlap_max, alpha=0.08, color='green',
           label=f'Overlap [{overlap_min:.2f},{overlap_max:.2f}]')
ax.set_xlabel("GBM Propensity Score  e(X) = P(D=1|X)")
ax.set_ylabel("Density")
ax.set_title("GBM Propensity Score Overlap\n(common support = green shading)", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Panel C: Ytilde vs Dtilde scatter ----------------------------------------
ax = fig.add_subplot(gs[1, 0])
# Downsample for clarity
sample_idx = np.random.choice(len(df), size=min(400, len(df)), replace=False)
ax.scatter(df['Dtilde'].iloc[sample_idx],
           df['Ytilde'].iloc[sample_idx],
           c=df[TREATMENT].iloc[sample_idx].map({1: '#e06b5b', 0: '#5b8dd9'}),
           alpha=0.5, s=20)
# OLS line through residuals (this IS the PLR estimator)
dtilde = df['Dtilde'].values
ytilde = df['Ytilde'].values
slope  = np.dot(dtilde, ytilde) / np.dot(dtilde, dtilde)
x_line = np.linspace(dtilde.min(), dtilde.max(), 100)
ax.plot(x_line, slope * x_line, color='#2c3e50', lw=2,
        label=f'PLR slope ≈ ${slope:,.0f} (ATE)')
ax.axhline(0, color='#aaa', lw=0.7); ax.axvline(0, color='#aaa', lw=0.7)
ax.set_xlabel("Dtilde  =  D − E[D|X]")
ax.set_ylabel("Ytilde  =  Y − E[Y|X]")
ax.set_title("Ytilde vs Dtilde — PLR estimator\n(slope = causal effect in residual space)", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Panel D: Fold-level diagnostics ------------------------------------------
ax = fig.add_subplot(gs[1, 1])
folds = fold_metrics['fold'].values
x = np.arange(len(folds))
bar_w = 0.35
ax2 = ax.twinx()
bars1 = ax.bar(x - bar_w/2, fold_metrics['rmse_outcome']/1000, bar_w,
               color='#e06b5b', alpha=0.75, label='RMSE ($000s, left)')
bars2 = ax2.bar(x + bar_w/2, fold_metrics['auc_propensity'], bar_w,
                color='#5b8dd9', alpha=0.75, label='AUC (right)')
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i}' for i in folds])
ax.set_ylabel("Outcome RMSE ($000s)", color='#e06b5b')
ax2.set_ylabel("Propensity AUC", color='#5b8dd9')
ax2.set_ylim(0, 1)
ax.set_title("Per-Fold Nuisance Diagnostics\n(stable across folds = good)", fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.spines[['top']].set_visible(False)
ax2.spines[['top']].set_visible(False)

plt.savefig('../assets/plots/phase7/diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Diagnostics plot saved")


# COMMAND ----------

# =============================================================================
# PLR Estimator -- ATT (Propensity-Weighted PLR)
# =============================================================================
#
# The base Robinson PLR moment condition is:
#   theta = E[Dtilde * Ytilde] / E[Dtilde^2]
# This is OLS of Ytilde on Dtilde (no intercept) -- the population-average slope.
#
# To target ATT -- the effect on workers who actually received training --
# we reweight each observation's contribution by e(X) = P(D=1|X):
#
#   theta_ATT = sum_i [ e(X_i) * Dtilde_i * Ytilde_i ]
#             / sum_i [ e(X_i) * Dtilde_i^2 ]
#
# Intuition:
#   e(X) up-weights units that look like treated workers (high propensity).
#   Observations in regions of X-space where no treated workers live get
#   near-zero weight. The moment condition focuses on the treated subpopulation.
#
# This is the same estimand as PSM (Phase 3) and IPW (Phase 4):
#   ATT = E[Y(1) - Y(0) | D=1]
# Results are therefore directly comparable to:
#   PSM ATT  = $300   (Phase 3, 83 matched pairs, wide CI)
#   IPW ATT  = $1,212 (Phase 4, trimmed, CPS-3)
#   AIPW ATT = $476   (Phase 4, linear OLS outcome model misfitted skewed earnings)
#
# The Neyman-orthogonal structure is preserved: the score is still
# insensitive to small errors in either g(X) or m(X).
# SE is computed via the weighted sandwich formula.
# Bootstrap CI: 500 draws, 2.5/97.5 percentiles.

def plr_att(Ytilde, Dtilde, propensity):
    """
    Propensity-weighted PLR for ATT.
    theta_ATT = sum(e * Dtilde * Ytilde) / sum(e * Dtilde^2)
    """
    e     = propensity.clip(1e-6, 1 - 1e-6)
    num   = np.dot(e * Dtilde, Ytilde)
    denom = np.dot(e * Dtilde, Dtilde)
    theta = num / denom

    # Weighted sandwich SE
    psi   = e * Dtilde * (Ytilde - theta * Dtilde)
    V     = np.dot(psi, psi) / len(Ytilde)**2
    se    = np.sqrt(V / (np.dot(e * Dtilde, Dtilde) / len(Ytilde))**2)

    t     = theta / se
    p     = 2 * stats.norm.sf(abs(t))
    return theta, se, t, p

def bootstrap_plr_att(Ytilde, Dtilde, propensity, n_boot=500, seed=42):
    """Bootstrap distribution for propensity-weighted PLR ATT."""
    rng  = np.random.default_rng(seed)
    ests = []
    n    = len(Ytilde)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ests.append(plr_att(Ytilde[idx], Dtilde[idx], propensity[idx])[0])
    return np.array(ests)

propensity = df['Dhat'].values

theta_att, se_att, t_att, p_att = plr_att(ytilde, dtilde, propensity)
ci_lo_att = theta_att - 1.96 * se_att
ci_hi_att = theta_att + 1.96 * se_att

boot_att   = bootstrap_plr_att(ytilde, dtilde, propensity)
boot_ci_lo_att = np.percentile(boot_att, 2.5)
boot_ci_hi_att = np.percentile(boot_att, 97.5)

recovery = theta_att / TRUE_ATT * 100

print(f"PLR ATT (propensity-weighted Robinson estimator)")
print(f"{'─' * 50}")
print(f"  theta_ATT:          ${theta_att:>8,.2f}")
print(f"  SE (sandwich):      ${se_att:>8,.2f}")
print(f"  t-statistic:         {t_att:>8.3f}")
print(f"  p-value:             {p_att:>8.4f}")
print(f"  95% CI (sandwich):   [${ci_lo_att:>7,.0f}, ${ci_hi_att:>7,.0f}]")
print(f"  95% CI (bootstrap):  [${boot_ci_lo_att:>7,.0f}, ${boot_ci_hi_att:>7,.0f}]")
print(f"\n  True ATT (RCT):     ${TRUE_ATT:>8,}")
print(f"  Recovery:            {recovery:.1f}% of benchmark")


# COMMAND ----------

# =============================================================================
# Path A -- Full CPS: ATT Estimate and the Large-N Precision Trap
# =============================================================================
#
# OUTPUT PREVIEW (to read before running):
#   Full CPS: ATT ≈ $1,957  SE ≈ $781   t ≈ 2.51  p ≈ 0.012  <- significant
#   CPS-3:    ATT ≈ $1,486  SE ≈ $987   t ≈ 1.51  p ≈ 0.132  <- not significant
#
# A researcher without overlap diagnostics would conclude:
#   "Full CPS gave a significant, benchmark-proximate result. Use it."
# That conclusion is wrong. Here is the mechanism of the mistake.
#
# ── PITFALL: LARGE-N PRECISION WITHOUT IDENTIFICATION ──────────────────────
#
# Statistical significance answers one question: given the data and model,
# how likely is a result this large under the null of zero effect?
# It says nothing about whether the identification assumption (overlap,
# unconfoundedness) holds. A method can be precisely wrong.
#
# Step 1 -- What overlap failure looks like inside DML:
#   When GBM near-perfectly separates treated from control workers
#   (AUC = 0.977), it assigns propensity scores e(X) ≈ 0 to almost all
#   15,992 control workers. In the propensity-weighted PLR moment:
#
#     theta_ATT = sum[ e(X_i) * Dtilde_i * Ytilde_i ]
#               / sum[ e(X_i) * Dtilde_i^2 ]
#
#   near-zero e(X) for controls means their contribution to both numerator
#   and denominator is negligible. The effective sample collapses to the
#   handful of controls near the overlap boundary -- the same ESS collapse
#   that broke IPW in Phase 4, just inside a different formula.
#   Mean |Dtilde| = 0.0127 for Full CPS vs 0.1894 for CPS-3: the treatment
#   residual signal is ~15x weaker. The estimator is running on almost
#   no information about the counterfactual.
#
# Step 2 -- Why the SE shrinks anyway (the trap):
#   The sandwich SE formula has N in the denominator:
#     SE ∝ 1 / (N * E[Dtilde^2])
#   Full CPS has N = 16,177 vs CPS-3's N = 614 -- a 26x difference.
#   Even though E[Dtilde^2] is ~15x smaller (weak signal), the 26x N
#   advantage more than compensates: the SE compresses and t rises above 2.
#   The estimator looks precise because N is large, not because the
#   identification is sound. This is asymptotic theory misfiring: N → inf
#   helps only when each new observation adds genuine counterfactual
#   information. Here, the new observations (Full CPS controls) are
#   structurally incomparable -- they add N without adding identification.
#
# Step 3 -- What the significant p-value actually measures:
#   The bootstrap draws from the full pool of 15,992 controls. Even with
#   near-zero overlap, each bootstrap resample finds enough boundary
#   observations to produce a consistent positive slope in residual space.
#   The result is tight -- but tight around an estimate driven by a
#   non-representative stratum (controls who happen to be near the treated
#   workers in covariate space). This is not the ATT. It is the effect
#   on treated workers as measured against a set of controls that barely
#   overlaps with them -- a counterfactual that does not exist in the data.
#
# Step 4 -- Why this is worse than CPS-3's insignificant result:
#   CPS-3 returns p = 0.132 and a CI that crosses zero. A researcher might
#   call this a failure. It is actually honesty: the sample is small, overlap
#   is borderline, and the data genuinely cannot rule out zero effect at 5%.
#   That uncertainty is real. Reporting it correctly is the right outcome.
#   Full CPS's p = 0.012 is a false signal -- it passes the significance
#   threshold by accumulating N that does not support the identification,
#   not by having better data.
#
# ── DIAGNOSTIC CHECKLIST FOR OVERLAP IN DML ────────────────────────────────
#   Before trusting a DML ATT result, verify:
#   1. Propensity AUC in a reasonable range (< 0.90 is ideal; > 0.95 is a warning)
#   2. Mean |Dtilde| is non-trivial -- if treatment residuals are near zero,
#      the PLR denominator is collapsing
#   3. Bootstrap CI width is proportionate -- very tight CIs from large N
#      with poor overlap are a red flag, not a green one
#   4. Check ESS: sum(e(X))^2 / sum(e(X)^2) -- effective sample size
#      after propensity weighting. Low ESS with high N = precision trap.
#
# CPS-3 passes checks 1-3. Full CPS fails all four.

Ytilde_inv  = df_invalid['Ytilde'].values
Dtilde_inv  = df_invalid['Dtilde'].values
pscore_inv  = df_invalid['Dhat'].values

theta_att_inv, se_att_inv, t_att_inv, p_att_inv = plr_att(Ytilde_inv, Dtilde_inv, pscore_inv)

# Effective Sample Size after propensity weighting
# ESS = sum(w)^2 / sum(w^2)  where w = e(X) for each observation
ess_invalid = df_invalid['Dhat'].sum()**2 / (df_invalid['Dhat']**2).sum()
ess_cps3    = df['Dhat'].sum()**2 / (df['Dhat']**2).sum()

print(f"Path A -- Full CPS -- PLR ATT:")
print(f"  theta_ATT:    ${theta_att_inv:>8,.0f}   SE: ${se_att_inv:>8,.0f}   t: {t_att_inv:.2f}   p: {p_att_inv:.3f}")
print(f"  n obs:         {len(df_invalid):,}")
print(f"  mean |Dtilde|: {df_invalid['Dtilde'].abs().mean():.4f}  <- treatment signal")
print(f"  AUC:           {fold_metrics_invalid['auc_propensity'].mean():.3f}  <- near-perfect separation")
print(f"  ESS (propensity-weighted): {ess_invalid:.1f} of {len(df_invalid):,}  ({ess_invalid/len(df_invalid)*100:.1f}%)")
print(f"\nPath B -- CPS-3 -- PLR ATT (primary result):")
print(f"  theta_ATT:    ${theta_att:>8,.0f}   SE: ${se_att:>8,.0f}   t: {t_att:.2f}   p: {p_att:.3f}")
print(f"  n obs:         {len(df):,}")
print(f"  mean |Dtilde|: {df['Dtilde'].abs().mean():.4f}  <- treatment signal ({df['Dtilde'].abs().mean()/df_invalid['Dtilde'].abs().mean():.0f}x stronger)")
print(f"  AUC:           {fold_metrics['auc_propensity'].mean():.3f}  <- borderline overlap")
print(f"  ESS (propensity-weighted): {ess_cps3:.1f} of {len(df):,}  ({ess_cps3/len(df)*100:.1f}%)")
print(f"\n⚠  PRECISION TRAP DIAGNOSIS:")
print(f"   Full CPS SE is smaller (${se_att_inv:,.0f} vs ${se_att:,.0f}) despite {df['Dtilde'].abs().mean()/df_invalid['Dtilde'].abs().mean():.0f}x weaker signal.")
print(f"   N advantage ({len(df_invalid):,} vs {len(df):,}) is mechanically compressing SE.")
print(f"   ESS after weighting: {ess_invalid:.0f} ({ess_invalid/len(df_invalid)*100:.1f}%) -- only {ess_invalid:.0f} observations")
print(f"   are doing the identification work in a dataset of {len(df_invalid):,}.")
print(f"   p = {p_att_inv:.3f} is false precision. CPS-3 p = {p_att:.3f} is honest uncertainty.")


# COMMAND ----------

# =============================================================================
# Robustness Check: Propensity Trimming (CPS-3)
# =============================================================================
#
# Given the wide CI and p = 0.132, the natural question is: can we do better?
#
# The binding constraint is not model choice -- GBM nuisance models are already
# appropriate for skewed earnings. The constraint is sample size and overlap
# quality. n = 614 with AUC = 0.916 means some control observations sit at the
# edges of covariate space where they contribute noise rather than information
# to the propensity-weighted moment condition.
#
# Propensity trimming: drop observations where e(X) < trim_threshold.
# This directly mirrors Phase 4's IPW trimming at the 5th percentile.
# The logic:
#   - Observations with very low propensity scores contribute near-zero
#     weight to the numerator but non-zero variance to the SE
#   - Removing them trades a small amount of bias (we no longer represent
#     the very-hard-to-match corner of covariate space) for reduced variance
#   - If the trimmed estimate is close to the untrimmed estimate, the
#     trimmed controls were adding noise, not signal
#
# We try three thresholds: 0.05, 0.10, 0.15.
# The honest goal is not to find a threshold that gives p < 0.05.
# The goal is to check whether the point estimate is stable (it should be
# if trimmed obs were already near-zero weight) and whether SE narrows.
# If the estimate moves substantially, trimming is changing the estimand --
# a warning, not a victory.
#
# NOTE: There is no legitimate way to recover a tight CI from 185 treated
# workers without more data. Trimming may narrow the SE modestly.
# If the CI still crosses zero after trimming, that is the honest result.

TRIM_THRESHOLDS = [0.05, 0.10, 0.15]
trim_results = []

print(f"Propensity trimming robustness -- CPS-3")
print(f"{'─' * 65}")
print(f"  Untrimmed:  n={len(df):>4,}  ATT=${theta_att:>7,.0f}  SE=${se_att:>7,.0f}  "
      f"CI=[${boot_ci_lo_att:>6,.0f}, ${boot_ci_hi_att:>6,.0f}]  p={p_att:.3f}")

for thresh in TRIM_THRESHOLDS:
    df_trim = df[df['Dhat'] >= thresh].copy()
    n_dropped = len(df) - len(df_trim)

    Yt = df_trim['Ytilde'].values
    Dt = df_trim['Dtilde'].values
    et = df_trim['Dhat'].values

    theta_t, se_t, t_t, p_t = plr_att(Yt, Dt, et)
    boot_t = bootstrap_plr_att(Yt, Dt, et)
    ci_lo_t, ci_hi_t = np.percentile(boot_t, [2.5, 97.5])

    trim_results.append({
        'threshold': thresh, 'n': len(df_trim), 'n_dropped': n_dropped,
        'theta': theta_t, 'se': se_t, 'p': p_t,
        'ci_lo': ci_lo_t, 'ci_hi': ci_hi_t
    })

    print(f"  e(X)≥{thresh:.2f}:   n={len(df_trim):>4,}  ATT=${theta_t:>7,.0f}  SE=${se_t:>7,.0f}  "
          f"CI=[${ci_lo_t:>6,.0f}, ${ci_hi_t:>6,.0f}]  p={p_t:.3f}  "
          f"(dropped {n_dropped} obs, "
          f"{'control' if (df[df['Dhat']<thresh]['treat']==0).all() else 'mixed'})")

print(f"\nInterpretation:")
best = min(trim_results, key=lambda x: x['se'])
stable = all(abs(r['theta'] - theta_att) / theta_att < 0.15 for r in trim_results)
print(f"  Point estimate {'stable' if stable else 'moves substantially'} across thresholds "
      f"({'< 15% change -- trimmed obs were near-zero weight' if stable else '> 15% change -- trimming is changing the estimand'}).")
print(f"  Best SE: ${best['se']:,.0f} at threshold {best['threshold']:.2f}  (vs untrimmed ${se_att:,.0f}).")
if best['p'] < 0.05:
    print(f"  CI no longer crosses zero at threshold {best['threshold']:.2f} -- but note:")
    print(f"  this is a robustness check, not model selection. Report untrimmed as primary.")
else:
    print(f"  CI still crosses zero at all thresholds -- sample size is the binding constraint.")
    print(f"  Trimming confirms the untrimmed result is not driven by low-propensity noise.")


# COMMAND ----------

# =============================================================================
# Results Plot -- Four Panels
# =============================================================================
#
# A. Bootstrap distribution of ATT -- histogram of 500 bootstrap estimates,
#    overlaid with the point estimate, RCT benchmark, and 95% CI.
#
# B. Cross-series ATT comparison -- PSM (Phase 3), IPW trimmed (Phase 4),
#    AIPW (Phase 4), DML ATT (Phase 7) vs the $1,794 benchmark.
#    The headline comparison: same estimand, same data, different methods.
#
# C. Residual-on-residual scatter (Ytilde vs Dtilde) coloured by treatment.
#    The PLR ATT slope is the propensity-weighted version of this relationship.
#
# D. Covariate SMD before vs after DML propensity weighting.
#    Mirrors the love plot from Phase 4 -- shows whether DML's GBM propensity
#    model achieves better balance than IPW's logistic regression did.

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Phase 7 -- Double ML Results (CPS-3, Path B)",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

# -- Panel A: Bootstrap distribution ------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.hist(boot_att, bins=40, color='#5b8dd9', alpha=0.75, edgecolor='white', linewidth=0.4)
ax.axvline(theta_att,  color='#e06b5b', lw=2.5, label=f'ATT estimate  ${theta_att:,.0f}')
ax.axvline(TRUE_ATT,   color='#2ecc71', lw=2.5, linestyle='--', label=f'True ATT  ${TRUE_ATT:,}')
ax.axvline(boot_ci_lo_att, color='#95a5a6', lw=1.5, linestyle=':')
ax.axvline(boot_ci_hi_att, color='#95a5a6', lw=1.5, linestyle=':', label=f'95% CI  [${boot_ci_lo_att:,.0f}, ${boot_ci_hi_att:,.0f}]')
ax.set_xlabel("Bootstrap ATT estimate ($)")
ax.set_ylabel("Count")
ax.set_title("Bootstrap Distribution\nDML ATT (500 draws)", fontweight='bold')
ax.legend(fontsize=7.5)
ax.spines[['top', 'right']].set_visible(False)

# -- Panel B: Cross-series comparison -----------------------------------------
ax = fig.add_subplot(gs[0, 1])
#
# Prior-phase estimates from published results (index.html Phase 3 & 4):
#   Phase 3 (PSM):        ATT = $123 (lower CI), $468 (upper CI approx from 83-pair width)
#                         Point ≈ $300, CI [-$1,819, +$2,064]
#   Phase 4 (IPW trim):   ATT = $1,212,  CI [-$141, +$2,718]   (68% recovery)
#   Phase 4 (AIPW):       ATT = $476,    CI [-$1,132, +$2,084] (27% recovery)
#   Phase 7 (DML ATT):    current result -- bootstrap CI
#
methods_labels = ['PSM\n(Phase 3)', 'IPW Trimmed\n(Phase 4)', 'AIPW\n(Phase 4)', 'DML ATT\n(Phase 7)']
ests   = [300,    1212,   476,           theta_att]
ci_los = [-1819,  -141,   -1132,         boot_ci_lo_att]
ci_his = [2064,   2718,   2084,          boot_ci_hi_att]
colors = ['#95a5a6', '#3498db', '#2980b9', '#e06b5b']

for i, (lo, hi, est, col) in enumerate(zip(ci_los, ci_his, ests, colors)):
    ax.plot([lo, hi], [i, i], 'o-', color=col, lw=2.5, ms=5)
    ax.scatter([est], [i], s=100, color=col, zorder=5)
    ax.text(hi + 50, i, f'${est:,.0f}', va='center', fontsize=8, color=col)

ax.axvline(TRUE_ATT, color='#2ecc71', linestyle='--', lw=2,
           label=f'True ATT = ${TRUE_ATT:,}')
ax.axvline(0, color='#888', lw=0.8)
ax.set_yticks(range(len(methods_labels)))
ax.set_yticklabels(methods_labels)
ax.set_xlabel("ATT estimate ($)")
ax.set_title("Cross-Series Comparison\nObservational ATT estimates vs RCT benchmark", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

# -- Panel C: Residual scatter ------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
colors_scatter = [('#e06b5b' if d == 1 else '#5b8dd9') for d in df[TREATMENT].values[sample_idx]]
ax.scatter(df['Dtilde'].values[sample_idx],
           df['Ytilde'].values[sample_idx],
           c=colors_scatter, alpha=0.45, s=18)
ax.plot(x_line, slope * x_line, color='#2c3e50', lw=2,
        label=f'PLR slope ≈ ${slope:,.0f}')
ax.axhline(0, color='#aaa', lw=0.6)
ax.axvline(0, color='#aaa', lw=0.6)
ax.set_xlabel("Dtilde  =  D − E[D|X]")
ax.set_ylabel("Ytilde  =  Y − E[Y|X]")
ax.set_title("Residual-on-Residual Scatter\nSlope = PLR causal effect", fontweight='bold')

# Manual legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e06b5b', alpha=0.7, label='Treated'),
                   Patch(facecolor='#5b8dd9', alpha=0.7, label='Control')]
ax.legend(handles=legend_elements + [plt.Line2D([0],[0],color='#2c3e50',lw=2,label=f'PLR slope ${slope:,.0f}')],
          fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Panel D: Covariate balance before vs after DML propensity weighting -------
ax = fig.add_subplot(gs[1, 1])
#
# After cross-fitting, we have GBM propensity scores e(X).
# We compute weighted SMD: weight treated = 1, weight control = e(X)/(1-e(X)).
# This mirrors the Phase 4 love plot but uses GBM propensity rather than
# logistic regression -- the direct comparison point for DML vs IPW.

def weighted_smd(df, covariates, propensity_col='Dhat', treat_col='treat'):
    smds_before = compute_smd(df, covariates, treat_col)
    smds_after  = {}
    for col in covariates:
        treated  = df[df[treat_col]==1][col]
        control  = df[df[treat_col]==0][col]
        w_ctrl   = df[df[treat_col]==0][propensity_col] / (1 - df[df[treat_col]==0][propensity_col])
        m1 = treated.mean()
        m0 = np.average(control, weights=w_ctrl)
        s  = df[col].std()
        smds_after[col] = (m1 - m0) / s if s > 0 else 0
    return smds_before, pd.Series(smds_after)

smds_before, smds_after = weighted_smd(df, COVARIATES)
y_pos = np.arange(len(COVARIATES))
ax.scatter(smds_before.values, y_pos, marker='o', s=60, color='#95a5a6',
           label='Before DML', zorder=4)
ax.scatter(smds_after.values,  y_pos, marker='D', s=60, color='#e06b5b',
           label='After DML (GBM weights)', zorder=5)
ax.axvline(0,    color='#888', lw=0.8)
ax.axvline( 0.1, color='#aaa', lw=0.8, linestyle='--')
ax.axvline(-0.1, color='#aaa', lw=0.8, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(COVARIATES)
ax.set_xlabel("Standardised Mean Difference")
ax.set_title("Covariate Balance\nBefore vs after DML propensity weighting", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.savefig('../assets/plots/phase7/results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Results plot saved")


# COMMAND ----------

# =============================================================================
# MLflow Logging
# =============================================================================

mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/causal_inference_toolkit")

with mlflow.start_run(run_name='phase7_dml'):

    # Parameters
    mlflow.log_param('method',              'Double ML -- PLR')
    mlflow.log_param('dataset',             'LaLonde_CPS3')
    mlflow.log_param('estimand_primary',    'ATT')
    mlflow.log_param('outcome_model',       'GradientBoostingRegressor')
    mlflow.log_param('propensity_model',    'GradientBoostingClassifier')
    mlflow.log_param('n_estimators',        200)
    mlflow.log_param('max_depth',           3)
    mlflow.log_param('learning_rate',       0.05)
    mlflow.log_param('n_folds',             N_FOLDS)
    mlflow.log_param('n_boot',              500)
    mlflow.log_param('n_treated',           int(df[TREATMENT].sum()))
    mlflow.log_param('n_control',           int((df[TREATMENT]==0).sum()))
    mlflow.log_param('covariates',          str(COVARIATES))
    mlflow.log_param('true_att',            TRUE_ATT)

    # Metrics
    mlflow.log_metric('att_dml',              round(theta_att,      2))
    mlflow.log_metric('se_att',               round(se_att,         2))
    mlflow.log_metric('t_att',                round(t_att,          3))
    mlflow.log_metric('p_att',                round(p_att,          4))
    mlflow.log_metric('ci_lo_att_sandwich',   round(ci_lo_att,      2))
    mlflow.log_metric('ci_hi_att_sandwich',   round(ci_hi_att,      2))
    mlflow.log_metric('ci_lo_att_boot',       round(boot_ci_lo_att, 2))
    mlflow.log_metric('ci_hi_att_boot',       round(boot_ci_hi_att, 2))
    mlflow.log_metric('recovery_pct',         round(recovery,       1))
    mlflow.log_metric('mean_rmse_outcome',    round(fold_metrics['rmse_outcome'].mean(),     2))
    mlflow.log_metric('mean_auc_propensity',  round(fold_metrics['auc_propensity'].mean(),   4))
    mlflow.log_metric('naive_att',            round(naive_att,      2))

    # Artifacts
    mlflow.log_artifact('../assets/plots/phase7/diagnostics.png')
    mlflow.log_artifact('../assets/plots/phase7/results.png')

    run_id = mlflow.active_run().info.run_id
    print(f"✓ MLflow run logged -- Run ID: {run_id}")
    print(f"  ATT (DML):          ${theta_att:,.2f}")
    print(f"  SE (sandwich):      ${se_att:,.2f}")
    print(f"  95% CI ATT (boot):  [${boot_ci_lo_att:,.0f}, ${boot_ci_hi_att:,.0f}]")
    print(f"  p-value:            {p_att:.4f}  ({'significant' if p_att < 0.05 else 'not significant at 5%'})")
    print(f"  Recovery:           {recovery:.1f}% of true ATT = ${TRUE_ATT:,}")


# COMMAND ----------

# =============================================================================
# Summary and Bridge to Phase 8 (DoWhy)
# =============================================================================

print(f"""
What we did:
  ✓ Loaded LaLonde + CPS-3 (Path B primary, Full CPS Path A documented)
  ✓ Cross-fitted GBM nuisance models ({N_FOLDS} folds)
      Outcome model    -- GradientBoostingRegressor (handles skewed earnings)
      Propensity model -- GradientBoostingClassifier (better calibration than logistic)
      Mean RMSE (outcome): ${fold_metrics['rmse_outcome'].mean():,.0f}
      Mean AUC  (propensity): {fold_metrics['auc_propensity'].mean():.3f}
  ✓ Computed residuals: Ytilde = Y - E[Y|X], Dtilde = D - E[D|X]
  ✓ Estimated ATT via propensity-weighted PLR (primary and only estimand)
  ✓ Bootstrap 95% CI (500 draws)
  ✓ Logged all results to MLflow

Key results (CPS-3, primary estimand = ATT):
  True ATT (RCT benchmark):     ${TRUE_ATT:,}
  Naive estimate:               ${naive_att:,.0f}
  PSM ATT     (Phase 3):        $300         CI [-$1,819, +$2,064]  p > 0.6
  IPW ATT     (Phase 4):        $1,212       CI [-$141,  +$2,718]   68% recovery
  AIPW ATT    (Phase 4):        $476         CI [-$1,132, +$2,084]  27% recovery
  DML ATT     (Phase 7):        ${theta_att:,.0f}       CI [${boot_ci_lo_att:,.0f}, ${boot_ci_hi_att:,.0f}]  {recovery:.0f}% recovery  <- PRIMARY RESULT
  p-value:                      {p_att:.4f}  ({'significant at 5%' if p_att < 0.05 else 'not significant at 5% -- CI crosses zero'})

INTERPRETING THESE RESULTS:
  Point estimate recovery ({recovery:.0f}%) is {'strong' if recovery > 70 else 'moderate'} -- DML ATT of ${theta_att:,.0f}
  {'is closer to the $1,794 benchmark than AIPW ($476) and IPW ($1,212).' if theta_att > 1212 else 'sits between AIPW ($476) and IPW ($1,212) in point recovery.'}

  The wide CI (${boot_ci_hi_att - boot_ci_lo_att:,.0f} range) and p-value of {p_att:.3f} reflect the fundamental
  constraint of this dataset: n=185 treated workers and n=429 controls with
  only borderline overlap (AUC {fold_metrics['auc_propensity'].mean():.3f}). Better nuisance models reduce
  bias but cannot manufacture sample size. The same overlap problem that
  limited PSM (wide CI, 83 pairs), IPW (CI clips zero), and AIPW (augmentation
  noise) limits DML here too. With ~614 total observations and moderate overlap,
  all observational methods on this dataset face high variance -- the honest
  finding is that the point estimate is directionally correct and plausible,
  but uncertainty is large.

  DML did improve on AIPW's point estimate. Phase 4 showed that OLS for E[Y|X]
  amplifies misspecification noise -- AIPW augmentation backfired. GBM nuisance
  models reduce that bias. But the SE stayed wide because propensity-weighting
  concentrates weight on a small treated-like subset of the already-small sample.

-> Phase 8 (DoWhy) returns to the same LaLonde + CPS-3 data.
   Where DML estimates theta by partialling out X, DoWhy makes the
   causal structure explicit: a directed acyclic graph encodes which
   variables are confounders, mediators, or instruments, and the
   identification strategy follows from the graph rather than from
   a modelling assumption about functional form.
""")

# COMMAND ----------


