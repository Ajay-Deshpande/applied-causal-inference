# Databricks notebook source
# Phase 4 -- Inverse Probability Weighting (IPW)
# Applied Causal Inference Series
#
# Problem:  Does job training increase earnings?
# Method:   Inverse Probability Weighting (IPW) + Augmented IPW (doubly-robust)
# Dataset:  LaLonde (1986) -- same treated workers as Phase 3, same two control groups
# Estimand: ATT -- Average Treatment Effect on the Treated (PRIMARY)
#           We keep all 185 treated workers (PSM discarded 102) and reweight
#           the control group to resemble the treated population.
#           ATT is directly comparable to the LaLonde RCT benchmark of $1,794.
# Benchmark: $1,794 (LaLonde RCT)
#
# Why ATT and not ATE?
#   The RCT gives us ATT = $1,794: the effect on workers who actually enrolled.
#   ATE would measure the effect if training were extended to the full CPS
#   population -- including workers who would never realistically enroll.
#   That's a different (and here unverifiable) question.
#   ATT is the right estimand for comparison with the benchmark.
#
# What Phase 3 (PSM) couldn't do that IPW does:
#   PSM discarded 102 of 185 treated workers (55%) because no control unit
#   was close enough under the caliper. The matched sample of 83 pairs was
#   too small to detect the $1,794 signal -- SE ~$990, CI spanning $4,000.
#
#   IPW keeps every observation. Instead of finding a match and discarding
#   workers without one, it assigns controls a weight proportional to how
#   'treated-like' they are: w = e(X) / (1 - e(X)).
#   Treated workers keep weight 1. No one is dropped.
#
# Two-path structure (mirrors Phase 3):
#   Path A -- Full CPS (15,992 controls): AUC ~1.0, near-zero overlap
#             IPW weights become extreme; ESS collapses. Documents failure.
#   Path B -- CPS-3  (   429 controls):  AUC ~0.87, borderline overlap
#             IPW feasible with trimming. Primary analysis.
#
# ATE is computed at the end as a secondary result with interpretation.


# COMMAND ----------

# MAGIC %pip install scipy statsmodels scikit-learn mlflow --quiet
# MAGIC dbutils.library.restartPython()
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
import statsmodels.api as sm
import mlflow
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

TRUE_ATT = 1794  # LaLonde RCT benchmark
print("\u2713 Imports complete")
print(f"  Benchmark ATT: ${TRUE_ATT:,}")


# COMMAND ----------

# =============================================================================
# Simulation Fallback
# =============================================================================
#
# If the NBER URLs are unavailable (no internet on cluster), fall back to
# simulated data calibrated to LaLonde's published summary statistics.
# Identical to Phase 3 -- one consistent fallback across the series.

TRUE_ATT = 1794

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
# Same data loading as Phase 3 -- identical treated group, two control groups.
# We build both combined DataFrames upfront so the two-path comparison is clean.
#
# df_invalid = treated + Full CPS (15,992 controls)  -> Path A: overlap failure
# df         = treated + CPS-3   (  429 controls)    -> Path B: primary analysis
#
# KEY CONTINUITY FROM PHASE 3:
#   Same NBER URLs, same COLUMNS definition, same COVARIATES.
#   IPW uses the same propensity model inputs as PSM -- any difference in
#   results is due to the weighting strategy, not the propensity model.

TREATED_URL     = "http://www.nber.org/~rdehejia/data/nswre74_treated.txt"
CONTROL_URL     = "http://www.nber.org/~rdehejia/data/cps_controls.txt"
CPS_CONTROL_URL = "http://www.nber.org/~rdehejia/data/cps3_controls.txt"
COLUMNS = ['treat', 'age', 'educ', 'black', 'hisp', 'married', 'nodegree', 're74', 're75', 're78']

try:
    df_treat    = pd.read_csv(TREATED_URL,     names=COLUMNS, sep='  ')
    df_ctrl     = pd.read_csv(CONTROL_URL,     names=COLUMNS, sep='  ')
    df_cps_ctrl = pd.read_csv(CPS_CONTROL_URL, names=COLUMNS, sep='  ')
    print("\u2713 Loaded from NBER website")
except Exception:
    df_treat, df_ctrl, df_cps_ctrl = get_simulated_lalonde()
    print("\u2713 Loaded from simulation (NBER unavailable)")

# Path A: Full CPS -- large, incomparable control group (overlap failure expected)
df_invalid = pd.concat([df_treat, df_ctrl], ignore_index=True)
df_invalid['treat'] = df_invalid['treat'].astype(float).astype(int)

# Path B: CPS-3 -- trimmed control group (primary IPW analysis)
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
# Helper Functions
# =============================================================================
#
# compute_smd:  Same SMD function as Phase 3. |SMD| < 0.1 is the target.
#
# compare_naive: Prints naive estimate + covariate SMD table. Same as Phase 3.
#
# effective_sample_size:
#   ESS = (sum_w)^2 / sum(w^2)
#   Measures information content of weighted sample as an equivalent
#   unweighted n. ESS << N means a few extreme weights dominate.
#   ESS < 30% of N is the standard warning threshold.
#
# compute_weights: Three ATE variants + two ATT variants.
#   HT (raw):    w = T/e + (1-T)/(1-e)              -- can explode near 0/1
#   Stabilized:  multiply by marginal P(T)/P(C)      -- mean ~1 per group
#   Trimmed:     clip stabilized at [5th,95th] pctil -- standard applied default
#   ATT raw:     treated=1, controls weighted by e/(1-e)
#   ATT trimmed: ATT with trimmed control weights
#
# weighted_smd: SMD computed on weighted means/variances (for post-IPW balance).

def compute_smd(df, covariates, treat_col='treat'):
    """Compute standardized mean difference for each covariate."""
    smds = {}
    for col in covariates:
        m1 = df[df[treat_col]==1][col].mean()
        m0 = df[df[treat_col]==0][col].mean()
        s  = df[col].std()
        smds[col] = ((m1 - m0) / s) if s > 0 else 0
    return pd.Series(smds)

def compare_naive(df, outcome, covariates, true_att, treatment_col='treat'):
    """Compute naive estimate and baseline SMD table."""
    naive = (df[df[treatment_col]==1][outcome].mean()
           - df[df[treatment_col]==0][outcome].mean())
    print(f"\nNaive ATE (raw mean difference): ${naive:,.0f}")
    print(f"True ATT  (benchmark):           ${true_att:,}")
    print(f"\nCovariate SMD before reweighting:")
    print(f"{'Covariate':<12} {'SMD':>8}  {'Balance':>12}")
    print("\u2500" * 36)
    smds = compute_smd(df, covariates, treatment_col)
    for cov, smd in smds.items():
        status = "\u2713 OK" if abs(smd) < 0.1 else ("\u26a0 Moderate" if abs(smd) < 0.2 else "\u2717 Imbalanced")
        print(f"{cov:<12} {smd:>8.3f}  {status:>12}")
    print(f"\nMean |SMD|: {smds.abs().mean():.3f}  |  Max |SMD|: {smds.abs().max():.3f}")
    return naive, smds

def effective_sample_size(weights):
    """ESS = (sum_w)^2 / sum(w^2). Equivalent unweighted sample size."""
    return weights.sum()**2 / (weights**2).sum()

def compute_weights(df, treatment='treat', pscore_col='pscore', trim_q=0.05):
    """
    Compute ATE and ATT weight variants. Returns df with new weight columns:
      w_ht, w_sw, w_sw_trim  (ATE)   |   w_att, w_att_trim  (ATT)
    """
    T  = df[treatment].values
    e  = df[pscore_col].values.clip(1e-6, 1 - 1e-6)  # avoid division by zero
    p1 = T.mean()        # marginal P(T=1)
    p0 = 1 - p1          # marginal P(T=0)

    # ATE weights
    w_ht       = T / e + (1 - T) / (1 - e)                  # Horvitz-Thompson
    w_sw       = T * p1 / e + (1 - T) * p0 / (1 - e)        # stabilized (Hajek)
    lo, hi     = np.quantile(w_sw, [trim_q, 1 - trim_q])
    w_sw_trim  = np.clip(w_sw, lo, hi)                       # trimmed

    # ATT weights -- treated units keep weight 1;
    # controls reweighted to 'look like' treated: w = e/(1-e)
    w_att      = T.astype(float) + (1 - T) * (e / (1 - e))
    lo_c, hi_c = np.quantile(w_att[T == 0], [trim_q, 1 - trim_q])
    w_att_trim = w_att.copy()
    w_att_trim[T == 0] = np.clip(w_att[T == 0], lo_c, hi_c)

    df = df.copy()
    df['w_ht']       = w_ht
    df['w_sw']       = w_sw
    df['w_sw_trim']  = w_sw_trim
    df['w_att']      = w_att
    df['w_att_trim'] = w_att_trim
    return df, (lo, hi)

def weighted_smd(df, covariates, treatment, weight_col):
    """Weighted SMD using weighted means and weighted variances."""
    rows = []
    for cov in covariates:
        t  = df[df[treatment]==1]
        c  = df[df[treatment]==0]
        mt = np.average(t[cov], weights=t[weight_col])
        mc = np.average(c[cov], weights=c[weight_col])
        vt = np.average((t[cov] - mt)**2, weights=t[weight_col])
        vc = np.average((c[cov] - mc)**2, weights=c[weight_col])
        ps = np.sqrt((vt + vc) / 2)
        rows.append({'covariate': cov, 'smd': abs((mt - mc) / ps) if ps > 0 else 0})
    return pd.DataFrame(rows).set_index('covariate')['smd']


# COMMAND ----------

# =============================================================================
# Estimate Propensity Scores
# =============================================================================
#
# Same logistic regression model as Phase 3 -- same covariates, same scaler.
# IPW and PSM share the propensity model; the difference is entirely in what
# we do with the scores afterwards (reweight vs match-and-discard).
#
# AUC INTERPRETATION FOR IPW:
#   Too high (>0.95) -> near-perfect separation -> IPW weights will be extreme.
#   PSM stopped at this diagnosis; IPW runs but ESS collapses.
#   CALIBRATION matters more than discrimination:
#   e(X) must truly equal P(T=1|X). Logistic regression is naturally calibrated.

def estimate_propensity(df, covariates, treatment_col='treat'):
    """Estimate propensity scores via logistic regression."""
    X = df[covariates].values
    y = df[treatment_col].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)

    df = df.copy()
    df['pscore'] = lr.predict_proba(X_scaled)[:, 1]

    auc = roc_auc_score(y, df['pscore'])

    print(f"\nLogistic regression fitted.")
    print(f"  AUC: {auc:.3f}")
    print(f"\n  NOTE: AUC should be moderate (0.6-0.85) for well-behaved IPW weights.")
    print(f"  AUC > 0.95 -> near-perfect separation -> weights extreme -> ESS collapses")
    print(f"  AUC < 0.55 -> covariates don't predict treatment -> check data")

    treated_ps  = df[df[treatment_col]==1]['pscore']
    control_ps  = df[df[treatment_col]==0]['pscore']
    overlap_min = max(treated_ps.min(), control_ps.min())
    overlap_max = min(treated_ps.max(), control_ps.max())

    print(f"\nPropensity score summary:")
    print(df.groupby(treatment_col)['pscore'].describe().round(3).to_string())
    print(f"\nCommon support region: [{overlap_min:.3f}, {overlap_max:.3f}]")

    if auc > 0.95:
        print(f"\n\u26a0  WARNING: AUC = {auc:.3f} -- near-perfect separation.")
        print(f"   IPW will run but weights will be extreme. ESS expected to collapse.")
        print(f"   Overlap failure is the fundamental issue, not the method.")

    return df, treated_ps, control_ps, overlap_min, overlap_max, auc


# COMMAND ----------

# =============================================================================
# Visualize Propensity Score Overlap
# =============================================================================
#
# Same diagnostic as Phase 3 -- identical purpose.
# Good overlap: distributions share a large common support region (green).
# Poor overlap: one distribution clusters near 0, the other near 1.
#               Most IPW weights will have near-zero denominators -> explode.
#
# The green shaded region is where reweighting is reliable.
# Outside this region, estimates are extrapolations.

def visualize_overlap(treated_ps, control_ps, overlap_min, overlap_max,
                      title_suffix='', save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    fig.suptitle(f"Phase 4 -- Propensity Score Overlap{title_suffix}",
                 fontsize=13, fontweight='bold')

    ax.hist(control_ps, bins=50, alpha=0.55, color='#5b8dd9',
            label=f'Control (n={len(control_ps):,})', density=True)
    ax.hist(treated_ps, bins=50, alpha=0.70, color='#e06b5b',
            label=f'Treated (n={len(treated_ps):,})', density=True)
    ax.axvline(overlap_min, color='#333', linestyle='--', linewidth=1.2,
               label=f'Common support [{overlap_min:.2f}, {overlap_max:.2f}]')
    ax.axvline(overlap_max, color='#333', linestyle='--', linewidth=1.2)
    ax.axvspan(overlap_min, overlap_max, alpha=0.08, color='green')
    ax.set_xlabel("Propensity Score  e(X) = P(T=1 | X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity score distributions -- overlap determines IPW feasibility",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\u2713 Overlap plot saved")
    plt.show()


# COMMAND ----------

# =============================================================================
# Path A -- Full CPS Controls: Overlap Failure
# =============================================================================
#
# Mirrors Phase 3's invalid path exactly.
# Full CPS workers (n=15,992) are demographically distant from treated workers.
# Propensity model achieves near-perfect separation (AUC ~1.0).
#
# WHAT THIS MEANS FOR IPW (different from PSM):
#   PSM: stopped -- couldn't find matches, flagged inapplicable.
#   IPW: runs, but weights explode. A treated worker with e(X)=0.01 gets
#        weight=100. One observation can dominate the entire estimate.
#        Even after trimming, the ESS collapses to a fraction of N.
#
# We run the full IPW pipeline on Full CPS to show this failure quantitatively
# rather than just stopping at the diagnostic.

naive_att_invalid, smds_before_invalid = compare_naive(
    df_invalid, OUTCOME, COVARIATES, TRUE_ATT)

df_invalid, t_ps_inv, c_ps_inv, ov_min_inv, ov_max_inv, auc_invalid = estimate_propensity(
    df_invalid, COVARIATES)

visualize_overlap(t_ps_inv, c_ps_inv, ov_min_inv, ov_max_inv,
                  title_suffix=" -- Full CPS (Overlap Failure)")

print(f"\n  \u2717 Full CPS: AUC = {auc_invalid:.3f} -- IPW will produce extreme weights.")
print(f"    The same fundamental problem PSM detected. IPW does not fix it;")
print(f"    it quantifies the damage rather than refusing to run.")


# COMMAND ----------

# =============================================================================
# Path B -- CPS-3 Controls: Primary IPW Analysis
# =============================================================================
#
# CPS-3 (n=429) is the trimmed control group from Phase 3 PSM.
# Same borderline AUC (~0.87) -- same overlap region -- same propensity model.
#
# IPW improves on PSM here by keeping all 185 treated workers instead of
# discarding 102. The effective sample size is larger and the CI is narrower.

naive_att, smds_before = compare_naive(df, OUTCOME, COVARIATES, TRUE_ATT)

df, treated_ps, control_ps, overlap_min, overlap_max, auc = estimate_propensity(
    df, COVARIATES)

visualize_overlap(treated_ps, control_ps, overlap_min, overlap_max,
                  title_suffix=" -- CPS-3 (Primary Analysis)",
                  save_path='../assets/plots/phase4/overlap.png')

print(f"\n  \u2713 CPS-3: AUC = {auc:.3f} -- borderline overlap, IPW feasible with trimming.")
print(f"    Proceeding with stabilized + trimmed weights as the primary estimator.")


# COMMAND ----------

# =============================================================================
# Compute IPW Weights -- Both Datasets
# =============================================================================
#
# We compute weights for both datasets to compare their behavior:
#   CPS-3:    moderate overlap -> well-behaved weights
#   Full CPS: near-zero overlap -> extreme weights even after trimming
#
# TRIM QUANTILE = 5th / 95th percentile (trim_q = 0.05)
# Standard choice in applied work. More aggressive trimming (10%) reduces
# variance further but introduces more bias. Less trimming (1%) is closer
# to raw HT but variance stays high.
#
# After this cell each DataFrame has five new weight columns:
#   w_ht, w_sw, w_sw_trim   (ATE)
#   w_att, w_att_trim        (ATT)

df,         trim_bounds      = compute_weights(df)
df_invalid, trim_bounds_inv  = compute_weights(df_invalid)

print("CPS-3 stabilized ATE weight summary:")
print(df.groupby('treat')[['w_ht', 'w_sw', 'w_sw_trim']].describe().round(3).to_string())
print(f"\nTrim bounds (CPS-3):    [{trim_bounds[0]:.3f}, {trim_bounds[1]:.3f}]")
print(f"Trim bounds (Full CPS): [{trim_bounds_inv[0]:.3f}, {trim_bounds_inv[1]:.3f}]")
print(f"\nFull CPS max raw HT weight: {df_invalid['w_ht'].max():,.1f}  <- extreme weight problem")


# COMMAND ----------

# =============================================================================
# Effective Sample Size (ESS) Diagnostic
# =============================================================================
#
# ESS = (sum_w)^2 / sum(w^2)
#
# Interpretation:
#   ESS = N          -> all weights equal (no confounding, reweighting not needed)
#   ESS << N         -> a few extreme weights dominate -> high variance
#   ESS < 30% of N   -> warning: estimate is effectively from very few observations
#
# This is IPW's analog of PSM's 'matched pairs count'.
# PSM quantified overlap cost through dropped observations (102 of 185).
# IPW quantifies it through ESS -- observations are present but not all
# contributing meaningful information.

print(f"{'Dataset':<12}  {'Estimator':<22}  {'N':>6}  {'ESS':>8}  {'ESS %':>7}")
print("\u2500" * 62)

for ds_name, dset in [('CPS-3', df), ('Full CPS', df_invalid)]:
    N = len(dset)
    for label, col in [('HT (raw)', 'w_ht'), ('Stabilized', 'w_sw'), ('Trimmed', 'w_sw_trim')]:
        ess = effective_sample_size(dset[col])
        pct = ess / N * 100
        flag = "  \u26a0" if pct < 30 else ""
        print(f"{ds_name:<12}  {label:<22}  {N:>6,}  {ess:>8,.0f}  {pct:>6.1f}%{flag}")
    print()

print("Rule of thumb: ESS < 30% of N is a warning sign.")
print("Full CPS ESS collapse is the quantitative fingerprint of near-zero overlap.")


# COMMAND ----------

# =============================================================================
# Weight Distribution Plot
# =============================================================================
#
# Visualize the effect of stabilization and trimming on the weight tails.
# A heavy right tail means a few observations are dominating the estimate.
# Trimming tames the tail -- the 2x2 grid shows this for both datasets.
#
# Reading the plot:
#   Top row (CPS-3):    trimming makes a moderate improvement
#   Bottom row (Full CPS): trimming tames the tail but extreme weights remain --
#                          the underlying overlap problem cannot be trimmed away

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Phase 4 -- IPW Weight Distributions: Effect of Trimming',
             fontsize=13, fontweight='bold')

for row_idx, (dset, ds_label) in enumerate([(df, 'CPS-3'), (df_invalid, 'Full CPS')]):
    for col_idx, (wcol, w_label) in enumerate([('w_sw',      'Stabilized (untrimmed)'),
                                               ('w_sw_trim', 'Stabilized + Trimmed')]):
        ax = axes[row_idx][col_idx]
        tw = dset.loc[dset['treat']==1, wcol]
        cw = dset.loc[dset['treat']==0, wcol]

        ax.hist(cw, bins=60, alpha=0.55, color='#5b8dd9', label='Control', density=True)
        ax.hist(tw, bins=60, alpha=0.70, color='#e06b5b', label='Treated', density=True)
        ax.axvline(dset[wcol].max(), color='#e74c3c', ls='--', lw=1.2,
                   label=f'Max = {dset[wcol].max():.1f}')

        ess_val = effective_sample_size(dset[wcol])
        ax.set_xlabel("Weight")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{ds_label} -- {w_label}\n"
            f"ESS = {ess_val:,.0f} ({ess_val/len(dset)*100:.0f}% of N={len(dset):,})",
            fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/plots/phase4/weight_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Weight distribution plot saved")


# COMMAND ----------

# =============================================================================
# Covariate Balance After Reweighting (Love Plot)
# =============================================================================
#
# After IPW reweighting, the pseudo-populations should be balanced.
# We compare raw SMD (same as Phase 3 before matching) against weighted SMD
# after applying trimmed stabilized weights.
#
# Target: |SMD| < 0.1 for all covariates after reweighting.
# If balance is still poor after IPW, the overlap problem is severe enough
# that no weighting scheme will fully adjust for confounding.
#
# NOTE ON COMPARISON WITH PSM:
#   Phase 3 Love plot: computed on MATCHED sample (83 pairs).
#   This Love plot:    computed on FULL sample (all 185 treated + all controls)
#                      with IPW weights applied.
#   IPW achieves balance without dropping a single observation.

smds_after_cps3  = weighted_smd(df,        COVARIATES, TREATMENT, 'w_sw_trim')
smds_after_full  = weighted_smd(df_invalid, COVARIATES, TREATMENT, 'w_sw_trim')

print(f"\n{'Covariate':<12} {'Before (CPS-3)':>16} {'After IPW':>10} {'Improved':>10}")
print("\u2500" * 50)
for cov in COVARIATES:
    before = smds_before[cov]
    after  = smds_after_cps3[cov]
    ok     = "\u2713 OK" if abs(after) < 0.1 else "\u26a0"
    imp    = "\u2713"    if abs(after) < abs(before) else "\u2717"
    print(f"{cov:<12} {abs(before):>16.3f} {abs(after):>10.3f} {imp:>5} {ok}")

print(f"\nMean |SMD| before (CPS-3): {smds_before.abs().mean():.3f}")
print(f"Mean |SMD| after IPW:       {smds_after_cps3.abs().mean():.3f}")
print(f"Reduction:                  {(1 - smds_after_cps3.abs().mean()/smds_before.abs().mean())*100:.1f}%")

# -- Love Plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phase 4 -- Love Plot: Covariate Balance Before vs After IPW',
             fontsize=13, fontweight='bold')

for ax, smds_after_ds, ds_label, smds_bef in [
    (axes[0], smds_after_cps3, 'CPS-3',    smds_before),
    (axes[1], smds_after_full, 'Full CPS', smds_before_invalid)
]:
    y_pos = np.arange(len(COVARIATES))
    ax.scatter(smds_bef.abs()[COVARIATES],    y_pos, color='#e06b5b', s=80,
               zorder=3, label='Before IPW', marker='o')
    ax.scatter(smds_after_ds[COVARIATES],      y_pos, color='#5b8dd9', s=80,
               zorder=3, label='After IPW (trimmed)', marker='D')
    for i, cov in enumerate(COVARIATES):
        ax.plot([smds_bef.abs()[cov], smds_after_ds[cov]], [i, i],
                color='#ccc', linewidth=1.5, zorder=2)
    ax.axvline(0.1, color='#333', linestyle='--', linewidth=1.2,
               label='|SMD| = 0.1 threshold')
    ax.axvline(0.0, color='#888', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COVARIATES)
    ax.set_xlabel("Absolute Standardized Mean Difference (SMD)")
    ax.set_title(f'Love Plot -- {ds_label}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/plots/phase4/love_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Love plot saved")


# COMMAND ----------

# =============================================================================
# ATT Estimation -- IPW (Primary Estimand)
# =============================================================================
#
# ATT weight formula:
#   Treated units:  weight = 1     (they ARE the population of interest)
#   Control units:  weight = e(X) / (1 - e(X))   (how 'treated-like' is this control?)
#
# The reweighted control group becomes a synthetic comparison population that
# mirrors the treated group's covariate distribution.
#
# Hajek (normalized) ATT estimator:
#   ATT = mean(Y | T=1)  -  sum(w_c * Y_c) / sum(w_c)
#   where w_c = e(X) / (1-e(X)) for controls, trimmed at [5th, 95th] pctile.
#
# Treated mean is just the raw mean -- weights are 1, normalization is trivial.
# All the action is on the control side.
#
# Bootstrap CIs (500 iterations). Same approach as Phase 3 PSM.
# Three variants: raw ATT, stabilized ATT, trimmed ATT.
# Trimmed is the primary result -- directly comparable to PSM ATT.

def ipw_att_hajek(df, outcome, treatment, weight_col):
    """Normalized IPW ATT estimator."""
    T  = df[treatment].values
    Y  = df[outcome].values
    w  = df[weight_col].values
    mu1 = Y[T == 1].mean()                                   # treated: weight=1
    mu0 = np.sum(w[T == 0] * Y[T == 0]) / np.sum(w[T == 0]) # reweighted controls
    return mu1 - mu0

def bootstrap_att_ci(df, outcome, treatment, weight_col, n_boot=500, seed=42):
    """Bootstrap 95% CI for IPW ATT."""
    rng  = np.random.default_rng(seed)
    ests = []
    for _ in range(n_boot):
        boot = df.sample(len(df), replace=True,
                         random_state=int(rng.integers(int(1e6))))
        ests.append(ipw_att_hajek(boot, outcome, treatment, weight_col))
    return np.percentile(ests, [2.5, 97.5])

print("Computing ATT bootstrap CIs (500 iterations each) -- may take ~30 seconds...")
print()

results = {}

# CPS-3: three ATT estimators
for label, wcol in [
    ('Raw',       'w_att'),
    ('Trimmed',   'w_att_trim'),
]:
    att = ipw_att_hajek(df, OUTCOME, TREATMENT, wcol)
    ci  = bootstrap_att_ci(df, OUTCOME, TREATMENT, wcol)
    results[f'CPS3_{label}'] = {'att': att, 'ci_lo': ci[0], 'ci_hi': ci[1]}
    print(f"CPS-3   {label:<20}  ATT = ${att:>7,.0f}   95% CI [${ci[0]:>7,.0f}, ${ci[1]:>7,.0f}]")

# Full CPS: trimmed only
att_full = ipw_att_hajek(df_invalid, OUTCOME, TREATMENT, 'w_att_trim')
ci_full  = bootstrap_att_ci(df_invalid, OUTCOME, TREATMENT, 'w_att_trim')
results['FullCPS_Trimmed'] = {'att': att_full, 'ci_lo': ci_full[0], 'ci_hi': ci_full[1]}
print(f"Full CPS {'Trimmed':<20}  ATT = ${att_full:>7,.0f}   95% CI [${ci_full[0]:>7,.0f}, ${ci_full[1]:>7,.0f}]")

print(f"\nBenchmark ATT (LaLonde RCT): ${TRUE_ATT:,}")
print(f"PSM ATT (Phase 3):           $123 - $468  (p > 0.6, n=83 pairs)")
print(f"\nIPW ATT uses all {int(df['treat'].sum())} treated workers -- PSM used 83.")


# COMMAND ----------

# =============================================================================
# Augmented IPW (AIPW) -- Doubly-Robust ATT Estimator
# =============================================================================
#
# AIPW for ATT (Robins, Rotnitzky, Zhao 1994):
#
#   AIPW_ATT = E[m1(X) - m0(X) | T=1]
#              + E[ T/P(T=1) * (Y - m1(X)) ]          <- augmentation: treated
#              - E[ (1-T)*e(X)/((1-e(X))*P(T=1)) * (Y - m0(X)) ]  <- controls
#
# Doubly robust: consistent if EITHER the propensity model OR the outcome
# regression is correctly specified -- not both.
#
# For ATT specifically:
#   The outcome model m0(X) must be good at predicting control outcomes.
#   The propensity model only needs to rank controls by treated-similarity.
#   Two independent lines of defense against misspecification.
#
# Cross-fitting (2-fold): outcome models trained on fold A, predictions on B.
# Same approach as any doubly-robust estimator -- prevents overfitting bias.

def aipw_att_estimate(df, outcome, treatment, covariates, pscore_col='pscore', n_folds=2):
    """Cross-fitted AIPW doubly-robust ATT with analytical SE."""
    df  = df.copy().reset_index(drop=True)
    T   = df[treatment].values
    Y   = df[outcome].values
    e   = df[pscore_col].values.clip(1e-6, 1 - 1e-6)
    X   = df[covariates].values
    p1  = T.mean()   # marginal P(T=1)

    m1_hat = np.zeros(len(df))
    m0_hat = np.zeros(len(df))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_tr, Y_tr, T_tr = X[train_idx], Y[train_idx], T[train_idx]
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X[test_idx])
        ols1 = LinearRegression().fit(X_tr_s[T_tr == 1], Y_tr[T_tr == 1])
        ols0 = LinearRegression().fit(X_tr_s[T_tr == 0], Y_tr[T_tr == 0])
        m1_hat[test_idx] = ols1.predict(X_te_s)
        m0_hat[test_idx] = ols0.predict(X_te_s)

    # ATT influence function scores
    # Treated arm: (Y - m1(X)) / P(T=1)  [correction for treated outcomes]
    # Control arm: e(X)/(1-e(X)) * (Y - m0(X)) / P(T=1)  [reweighted controls]
    phi = (m1_hat - m0_hat
           + T / p1 * (Y - m1_hat)
           - (1 - T) * e / ((1 - e) * p1) * (Y - m0_hat))

    att = phi.mean()
    se  = phi.std() / np.sqrt(len(df))
    ci  = (att - 1.96 * se, att + 1.96 * se)
    return att, ci, se

print("Computing AIPW ATT estimates...")
print()

att_aipw_cps3, ci_aipw_cps3, se_aipw_cps3 = aipw_att_estimate(
    df, OUTCOME, TREATMENT, COVARIATES)
results['CPS3_AIPW'] = {'att': att_aipw_cps3,
                         'ci_lo': ci_aipw_cps3[0], 'ci_hi': ci_aipw_cps3[1]}

att_aipw_full, ci_aipw_full, se_aipw_full = aipw_att_estimate(
    df_invalid, OUTCOME, TREATMENT, COVARIATES)
results['FullCPS_AIPW'] = {'att': att_aipw_full,
                            'ci_lo': ci_aipw_full[0], 'ci_hi': ci_aipw_full[1]}

print(f"CPS-3    AIPW ATT = ${att_aipw_cps3:>7,.0f}   SE = ${se_aipw_cps3:,.0f}"
      f"   95% CI [${ci_aipw_cps3[0]:>7,.0f}, ${ci_aipw_cps3[1]:>7,.0f}]")
print(f"Full CPS AIPW ATT = ${att_aipw_full:>7,.0f}   SE = ${se_aipw_full:,.0f}"
      f"   95% CI [${ci_aipw_full[0]:>7,.0f}, ${ci_aipw_full[1]:>7,.0f}]")
print(f"\nBenchmark ATT (LaLonde RCT): ${TRUE_ATT:,}")


# COMMAND ----------

# =============================================================================
# Full Results Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle('Phase 4 -- Inverse Probability Weighting Results (Primary: ATT)',
             fontsize=13, fontweight='bold')

# -- Plot A: ATT weight distribution (CPS-3, trimmed) -----------------------
ax = fig.add_subplot(gs[0, 0])
tw = df.loc[df['treat']==1, 'w_att_trim']
cw = df.loc[df['treat']==0, 'w_att_trim']
ax.hist(cw, bins=40, alpha=0.55, color='#5b8dd9', label='Control (reweighted)', density=True)
ax.axvline(1.0, color='#e06b5b', ls='--', lw=1.8,
           label='Treated (weight = 1)')
ax.set_xlabel("ATT weight  e(X)/(1-e(X))")
ax.set_ylabel("Density")
ax.set_title('ATT control weights after trimming\n(CPS-3 -- primary analysis)',
             fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

# -- Plot B: Overlap comparison (CPS-3 vs Full CPS) -------------------------
ax = fig.add_subplot(gs[0, 1])
ax.hist(control_ps, bins=40, alpha=0.5, color='#5b8dd9',
        label=f'CPS-3 control (n={len(control_ps):,})', density=True)
ax.hist(treated_ps, bins=40, alpha=0.7, color='#e06b5b',
        label=f'Treated (n={len(treated_ps):,})', density=True)
ax.hist(df_invalid.loc[df_invalid['treat']==0, 'pscore'],
        bins=40, alpha=0.25, color='#34495e', hatch='//',
        label=f'Full CPS control (n={(df_invalid["treat"]==0).sum():,})', density=True)
ax.set_xlabel("Propensity Score  e(X)")
ax.set_ylabel("Density")
ax.set_title('Overlap: CPS-3 vs Full CPS\n(Full CPS = near-zero overlap)', fontweight='bold')
ax.legend(fontsize=7)
ax.spines[['top', 'right']].set_visible(False)

# -- Plot C: ATT estimate comparison vs benchmark ---------------------------
ax = fig.add_subplot(gs[0, 2])
labels_c = ['True ATT\n(RCT)', 'Naive\ndiff', 'PSM\nPhase 3',
            'IPW ATT\nTrimmed', 'AIPW ATT\n(DR)']
ates_c   = [TRUE_ATT, naive_att, 300,
            results['CPS3_Trimmed']['att'], results['CPS3_AIPW']['att']]
colors_c = ['#2ecc71', '#e74c3c', '#95a5a6', '#3498db', '#2980b9']
bars     = ax.bar(labels_c, ates_c, color=colors_c, width=0.55, alpha=0.85)
for bar, val in zip(bars, ates_c):
    yoff = 60 if val >= 0 else -280
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
            f'${val:,.0f}', ha='center', fontsize=8, fontweight='bold')
ax.axhline(TRUE_ATT, color='#2ecc71', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'True ATT = ${TRUE_ATT:,}')
ax.axhline(0, color='#333', linewidth=0.8)
ax.set_title('ATT estimates vs benchmark\n(CPS-3 dataset)', fontweight='bold')
ax.set_ylabel("Estimate ($)")
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Plot D: ESS by estimator -----------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ess_vals = {
    'CPS-3':    [effective_sample_size(df['w_att']),
                 effective_sample_size(df['w_att_trim'])],
    'Full CPS': [effective_sample_size(df_invalid['w_att']),
                 effective_sample_size(df_invalid['w_att_trim'])],
}
x = np.arange(2)
w = 0.3
ax.bar(x - w/2, [v[0] for v in ess_vals.values()], w,
       label='Raw ATT',     color='#e74c3c', alpha=0.8)
ax.bar(x + w/2, [v[1] for v in ess_vals.values()], w,
       label='Trimmed ATT', color='#27ae60', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['CPS-3', 'Full CPS'])
ax.set_ylabel("Effective Sample Size (controls)")
ax.set_title('ESS by estimator\nFull CPS collapses even after trimming', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# -- Plot E: Love plot (CPS-3) -----------------------------------------------
ax = fig.add_subplot(gs[1, 1])
y_pos = np.arange(len(COVARIATES))
ax.scatter(smds_before.abs()[COVARIATES], y_pos, color='#e06b5b', s=70,
           zorder=3, label='Before IPW', marker='o')
ax.scatter(smds_after_cps3[COVARIATES],   y_pos, color='#5b8dd9', s=70,
           zorder=3, label='After IPW (trimmed)', marker='D')
for i, cov in enumerate(COVARIATES):
    ax.plot([smds_before.abs()[cov], smds_after_cps3[cov]], [i, i],
            color='#ccc', linewidth=1.5, zorder=2)
ax.axvline(0.1, color='#333', linestyle='--', linewidth=1.2, label='|SMD| = 0.1')
ax.set_yticks(y_pos)
ax.set_yticklabels(COVARIATES)
ax.set_xlabel("Absolute SMD")
ax.set_title('Covariate balance\nbefore vs after IPW (CPS-3)', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

# -- Plot F: CI comparison (IPW ATT vs PSM ATT) ------------------------------
ax = fig.add_subplot(gs[1, 2])
methods_ci = ['PSM ATT\n(Phase 3)', 'IPW ATT\nTrimmed', 'AIPW ATT\n(DR)']
ci_los     = [-1819, results['CPS3_Trimmed']['ci_lo'], results['CPS3_AIPW']['ci_lo']]
ci_his     = [2064,  results['CPS3_Trimmed']['ci_hi'], results['CPS3_AIPW']['ci_hi']]
ates_ci    = [300,   results['CPS3_Trimmed']['att'],   results['CPS3_AIPW']['att']]
for i, (lo, hi, est) in enumerate(zip(ci_los, ci_his, ates_ci)):
    col = ['#95a5a6', '#3498db', '#2980b9'][i]
    ax.plot([lo, hi], [i, i], 'o-', color=col, lw=2.5, ms=5)
    ax.scatter([est], [i], s=90, color=col, zorder=5)
ax.axvline(TRUE_ATT, color='#2ecc71', linestyle='--', lw=2,
           label=f'True ATT = ${TRUE_ATT:,}')
ax.axvline(0, color='#888', lw=0.8)
ax.set_yticks(range(3))
ax.set_yticklabels(methods_ci)
ax.set_xlabel("ATT estimate ($)")
ax.set_title('95% CI comparison\nIPW ATT vs PSM ATT (Phase 3)', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.savefig('../assets/plots/phase4/results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Results plot saved")


# COMMAND ----------

# =============================================================================
# MLflow Logging
# =============================================================================

mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/causal_inference_toolkit")

with mlflow.start_run(run_name='phase4_ipw'):

    # Parameters
    mlflow.log_param('method',         'IPW + AIPW')
    mlflow.log_param('dataset',        'LaLonde_CPS3')
    mlflow.log_param('estimand',       'ATT')
    mlflow.log_param('ps_model',       'LogisticRegression')
    mlflow.log_param('trim_quantile',  0.05)
    mlflow.log_param('n_treated',      int(df['treat'].sum()))
    mlflow.log_param('n_control',      int((df['treat']==0).sum()))
    mlflow.log_param('n_boot',         500)
    mlflow.log_param('aipw_folds',     2)
    mlflow.log_param('covariates',     str(COVARIATES))
    mlflow.log_param('true_att',       TRUE_ATT)

    # Metrics
    mlflow.log_metric('ps_auc_cps3',          round(auc,                                                     4))
    mlflow.log_metric('ess_att_raw',           round(effective_sample_size(df['w_att']),                      1))
    mlflow.log_metric('ess_att_trimmed',       round(effective_sample_size(df['w_att_trim']),                 1))
    mlflow.log_metric('att_raw',               round(results['CPS3_Raw']['att'],                              2))
    mlflow.log_metric('att_trimmed',           round(results['CPS3_Trimmed']['att'],                          2))
    mlflow.log_metric('att_aipw',              round(results['CPS3_AIPW']['att'],                             2))
    mlflow.log_metric('ci_lo_trimmed',         round(results['CPS3_Trimmed']['ci_lo'],                        2))
    mlflow.log_metric('ci_hi_trimmed',         round(results['CPS3_Trimmed']['ci_hi'],                        2))
    mlflow.log_metric('ci_lo_aipw',            round(results['CPS3_AIPW']['ci_lo'],                           2))
    mlflow.log_metric('ci_hi_aipw',            round(results['CPS3_AIPW']['ci_hi'],                           2))
    mlflow.log_metric('smd_mean_before',       round(smds_before.abs().mean(),                                4))
    mlflow.log_metric('smd_mean_after_ipw',    round(smds_after_cps3.abs().mean(),                            4))
    mlflow.log_metric('smd_reduction_pct',
                      round((1 - smds_after_cps3.abs().mean()/smds_before.abs().mean())*100,                  1))
    mlflow.log_metric('naive_att',             round(naive_att,                                               2))
    mlflow.log_metric('recovery_pct_trimmed',  round(results['CPS3_Trimmed']['att']/TRUE_ATT*100,             1))
    mlflow.log_metric('recovery_pct_aipw',     round(results['CPS3_AIPW']['att']   /TRUE_ATT*100,             1))

    # Artifacts
    mlflow.log_artifact('../assets/plots/phase4/overlap.png')
    mlflow.log_artifact('../assets/plots/phase4/weight_distributions.png')
    mlflow.log_artifact('../assets/plots/phase4/love_plot.png')
    mlflow.log_artifact('../assets/plots/phase4/results.png')

    run_id = mlflow.active_run().info.run_id
    print(f"\n\u2713 MLflow run logged -- Run ID: {run_id}")
    print(f"  ATT (trimmed IPW):  ${results['CPS3_Trimmed']['att']:,.2f}")
    print(f"  ATT (AIPW):         ${results['CPS3_AIPW']['att']:,.2f}")
    print(f"  95% CI (AIPW):      [${results['CPS3_AIPW']['ci_lo']:,.0f}, ${results['CPS3_AIPW']['ci_hi']:,.0f}]")
    print(f"  SMD reduction:      {(1 - smds_after_cps3.abs().mean()/smds_before.abs().mean())*100:.1f}%")
    print(f"  ESS (ATT trimmed):  {effective_sample_size(df['w_att_trim']):,.0f} "
          f"({effective_sample_size(df['w_att_trim'])/len(df)*100:.0f}% of N={len(df):,})")
    print(f"  Recovery (AIPW):    {results['CPS3_AIPW']['att']/TRUE_ATT*100:.1f}% of true ATT")


# COMMAND ----------

# =============================================================================
# Summary and Bridge to Phase 5 (Synthetic Control)
# =============================================================================

print(f"""
What we did:
  \u2713 Established baseline imbalance (mean |SMD| = {smds_before.abs().mean():.3f})
  \u2713 Estimated propensity scores via logistic regression
      CPS-3 AUC = {auc:.3f}  (borderline -- IPW feasible with trimming)
      Full CPS AUC = {auc_invalid:.3f}  (near-perfect separation -- weights extreme)
  \u2713 Visualized overlap -- two-path comparison: Full CPS fail vs CPS-3 feasible
  \u2713 Computed ATT weights: w = e(X)/(1-e(X)) for controls, 1 for treated
  \u2713 Computed ESS -- Full CPS collapse quantified numerically
  \u2713 Built Love plot -- weighted covariate balance on full sample (no discards)
  \u2713 Estimated ATT via Hajek IPW with bootstrap CIs  <- PRIMARY RESULT
  \u2713 Estimated ATT via cross-fitted AIPW (doubly-robust)  <- PRIMARY RESULT
  \u2713 Computed ATE as secondary result (see next cell)
  \u2713 Logged all results to MLflow

Key results (CPS-3, primary estimand = ATT):
  True ATT (RCT):         ${TRUE_ATT:,}
  Naive estimate:         ${naive_att:,.0f}    <- wrong sign
  PSM ATT (Phase 3):      $123 - $468           <- 83 pairs, p > 0.6
  IPW ATT trimmed:        ${results['CPS3_Trimmed']['att']:,.0f}
  AIPW ATT (doubly-robust):{results['CPS3_AIPW']['att']:,.0f}
  95% CI (AIPW):          [${results['CPS3_AIPW']['ci_lo']:,.0f}, ${results['CPS3_AIPW']['ci_hi']:,.0f}]
  Mean |SMD| before:      {smds_before.abs().mean():.3f}
  Mean |SMD| after IPW:   {smds_after_cps3.abs().mean():.3f}
  ESS (ATT trimmed):      {effective_sample_size(df['w_att_trim']):,.0f} of {len(df):,} ({effective_sample_size(df['w_att_trim'])/len(df)*100:.0f}%)

VERDICT: IPW improved on PSM by keeping all 185 treated workers.
  All estimates are directly comparable to the $1,794 RCT benchmark.
  AIPW's doubly-robust property provides additional protection against
  propensity model misspecification.

WHY IPW STILL STRUGGLES ON FULL CPS:
  1. AUC ~1.0 -> near-zero overlap -> ATT weights e(X)/(1-e(X)) explode
  2. ESS collapses to <5% of N even after trimming
  3. The fundamental problem is the data, not the method.
     PSM and IPW fail on the same datasets for the same root reason: overlap.

-> Phase 5 (Synthetic Control) abandons individual-level reweighting entirely.
   It works at the aggregate level: one treated unit (a state), a panel of
   donor states, and a weighted combination that matches the pre-treatment
   trajectory. Designed for small N, large T -- the opposite regime from LaLonde.
""")


# COMMAND ----------

# =============================================================================
# ATE -- Secondary Result (For Completeness)
# =============================================================================
#
# ATE is the canonical estimand for IPW in the survey sampling literature
# (Horvitz-Thompson, 1952). The historical reason: survey methods were designed
# to estimate population-level quantities, and the HT weight T/e(X) was derived
# to correct for unequal selection probabilities across the full population.
# That framing carried into causal inference -- hence 'canonical'.
#
# What ATE answers here:
#   'What would happen to earnings if job training were offered to everyone --
#    including CPS workers who would never realistically enroll?'
#
# Why we can't benchmark it:
#   The LaLonde RCT randomized among workers who applied for NSW training.
#   The RCT's $1,794 is an ATT: the effect on those who enrolled.
#   ATE includes a hypothetical effect on CPS workers who have different
#   earnings histories, demographics, and labor market attachment.
#   We have no ground truth to compare it against.
#
# What ATE can still tell us:
#   If ATE >> ATT: the training program would be even more valuable if extended
#                  to the broader population (positive spillover to new groups).
#   If ATE << ATT: the program works best for the people who self-select into it;
#                  expanding eligibility may dilute the effect.
#   In the LaLonde context, this comparison is speculative -- but it is a
#   legitimate policy question in other settings.

def ipw_ate_hajek(df, outcome, treatment, weight_col):
    """Normalized (Hajek) IPW ATE."""
    T  = df[treatment].values
    Y  = df[outcome].values
    w  = df[weight_col].values
    mu1 = np.sum(w * T * Y)       / np.sum(w * T)
    mu0 = np.sum(w * (1 - T) * Y) / np.sum(w * (1 - T))
    return mu1 - mu0

def bootstrap_ate_ci(df, outcome, treatment, weight_col, n_boot=500, seed=99):
    rng  = np.random.default_rng(seed)
    ests = []
    for _ in range(n_boot):
        boot = df.sample(len(df), replace=True,
                         random_state=int(rng.integers(int(1e6))))
        ests.append(ipw_ate_hajek(boot, outcome, treatment, weight_col))
    return np.percentile(ests, [2.5, 97.5])

print("ATE (secondary) -- CPS-3 dataset:")
print()

for label, wcol in [('Raw', 'w_sw'), ('Trimmed', 'w_sw_trim')]:
    ate = ipw_ate_hajek(df, OUTCOME, TREATMENT, wcol)
    ci  = bootstrap_ate_ci(df, OUTCOME, TREATMENT, wcol)
    print(f"  ATE {label:<10}  ${ate:>7,.0f}   95% CI [${ci[0]:>7,.0f}, ${ci[1]:>7,.0f}]")

print(f"\n  ATT trimmed (primary): ${results['CPS3_Trimmed']['att']:,.0f}")
print(f"  ATT AIPW   (primary): ${results['CPS3_AIPW']['att']:,.0f}")
print(f"\n  NOTE: ATE and ATT cannot be compared to the same benchmark.")
print(f"  The LaLonde $1,794 is an ATT. ATE is reported for completeness only.")
print(f"  In a setting with a known population-level estimand, ATE would be the headline.")

