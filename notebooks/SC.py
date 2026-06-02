# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# =============================================================================
# Phase 5 -- Synthetic Control
# Applied Causal Inference Series
#
# Problem:  Did a state-level policy intervention increase per-capita income?
# Method:   Synthetic Control Method (Abadie, Diamond, Hainmueller 2010)
# Dataset:  Simulated state panel -- 1 treated state, 20 donor states,
#           T=30 periods (treatment at t=21, i.e. 20 pre, 10 post)
# Estimand: ATT -- the effect on the treated state post-intervention
# Benchmark: known true effect embedded in simulation = $1,500
#
# Why Synthetic Control after IPW?
#   IPW, PSM, and AIPW are designed for settings with many treated and
#   control units. Synthetic Control is designed for the opposite regime:
#
#     N_treated = 1  (one state, country, city, or firm)
#     N_donors  = small (potential comparison units)
#     T         = large (many pre-treatment periods)
#
#   Instead of matching individuals on covariates, Synthetic Control
#   matches the treated unit's pre-treatment outcome trajectory using a
#   weighted combination of donor units.
#
#   If the synthetic unit closely reproduces the treated unit before
#   treatment, it serves as a data-driven counterfactual:
#
#     "What would have happened without the intervention?"
#
# Key steps:
#   1. Simulate panel data with a known treatment effect
#   2. Choose donor weights to minimize pre-treatment prediction error
#   3. Estimate ATT: treated outcome minus synthetic control (post-period)
#   4. Check pre-treatment fit using RMSPE (lower is better)
#   5. Placebo-in-time: assign treatment earlier and test for false effects
#   6. Placebo-in-space: reassign treatment to donor states and compare gaps
#   7. Inference: compare post/pre RMSPE ratios against placebo distributions

# COMMAND ----------

# MAGIC %pip install scipy statsmodels mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import mlflow
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

TRUE_EFFECT = 1500   # known true ATT embedded in simulation
T_TOTAL     = 30     # total time periods
T_TREAT     = 21     # first treated period (1-indexed); 20 pre, 10 post
N_DONORS    = 20     # donor states
print("\u2713 Imports complete")
print(f"  True ATT: ${TRUE_EFFECT:,}  |  Pre-periods: {T_TREAT-1}  |  Post-periods: {T_TOTAL-T_TREAT+1}")

# COMMAND ----------

# =============================================================================
# Simulate State Panel Data
# =============================================================================
#
# We simulate a panel of 21 states (1 treated + 20 donors) over 30 periods.
# The treated state (state_0) receives a policy at period 21.
#
# Data generating process:
#   - Each state has a fixed effect (baseline income level)
#   - All states share a common time trend (macroeconomic conditions)
#   - Each state has idiosyncratic shocks
#   - The treated state gets +TRUE_EFFECT per period after treatment
#
# Why simulate instead of real data?
#   Synthetic Control requires a single treated unit with a long pre-period.
#   The classic real example is California's Proposition 99 (tobacco control)
#   or the Basque Country terrorism study. We simulate so we can embed a
#   known true effect and evaluate the method's recovery precisely --
#   the same principle as using the LaLonde RCT as a benchmark.

def simulate_panel(n_donors=20, t_total=30, t_treat=21, true_effect=1500,
                   seed=42):
    """
    Simulate a balanced state panel with one treated unit.
    Returns a wide-format DataFrame: rows = periods, cols = states.
    """
    rng = np.random.default_rng(seed)

    # State fixed effects (baseline income levels)
    # Donor states are drawn uniformly between $15k-$30k to create
    # realistic cross-state variation while keeping states comparable.
    # The treated state is placed near the middle ($20k) so a reasonable
    # synthetic control can be constructed from donor combinations.
    fe_treated = 20_000
    fe_donors  = rng.uniform(15_000, 30_000, n_donors)

    # Common upward trend affecting all states
    # Total growth of $3k over 30 periods (~$100/period) creates a visible
    # secular trend without overwhelming the treatment effect.
    time_trend = np.linspace(0, 3_000, t_total)

    # Period-specific random shocks
    # N(0, 500²):
    #   Mean = 0     -> no systematic bias upward/downward
    #   SD   = 500   -> noticeable noise but smaller than the treatment
    #                  effect ($1,500), giving a signal-to-noise ratio of ~3.
    #
    # Shocks are independent across periods (not a random walk), so
    # temporary deviations do not persist into future periods.
    shock_treated = rng.normal(0, 500, t_total)
    shock_donors  = rng.normal(0, 500, (t_total, n_donors))

    # Treatment effect
    # Permanent increase of $1,500 beginning at t_treat.
    # Chosen to be large enough to detect but not so large that the
    # synthetic control problem becomes trivial.

    # Build outcome matrix
    # Treated state: fixed effect + trend + shock + effect post-treatment
    y_treated = (fe_treated + time_trend + shock_treated
                 + np.where(np.arange(t_total) >= t_treat - 1, true_effect, 0))

    # Donor states: fixed effect + trend + shock (no treatment)
    # Make fe_donors (1, n_donors) and time_trend (30, 1) and broadcast
    # to (30, n_donors)
    y_donors = fe_donors[np.newaxis, :] + time_trend[:, np.newaxis] + shock_donors

    periods = np.arange(1, t_total + 1)
    df = pd.DataFrame(y_donors,
                      columns=[f'donor_{i}' for i in range(1, n_donors + 1)],
                      index=periods)
    df.insert(0, 'treated', y_treated)
    df.index.name = 'period'
    return df

df = simulate_panel(N_DONORS, T_TOTAL, T_TREAT, TRUE_EFFECT)

DONOR_COLS  = [c for c in df.columns if c.startswith('donor_')]
PRE_MASK    = df.index < T_TREAT    # periods 1-20
POST_MASK   = df.index >= T_TREAT   # periods 21-30

print(f"Panel shape: {df.shape}  (periods x states)")
print(f"Pre-treatment periods:  {PRE_MASK.sum()}")
print(f"Post-treatment periods: {POST_MASK.sum()}")
print(f"\nTreated state pre-period income (mean): ${df.loc[PRE_MASK, 'treated'].mean():,.0f}")
print(f"Donor states  pre-period income (mean): ${df.loc[PRE_MASK, DONOR_COLS].mean().mean():,.0f}")
print(f"\nFirst 5 periods:")
df.head().round(0)


# COMMAND ----------

# =============================================================================
# Helper Functions
# =============================================================================
#
# synthetic_control_weights:
#   Solve the constrained QP to find donor weights W = (w_1, ..., w_J)
#   such that the pre-treatment synthetic outcome matches the treated outcome.
#
#   Objective:  minimize  ||Y_treated_pre - Y_donors_pre @ W||^2
#   Constraints: w_j >= 0 for all j  (non-negative weights)
#                sum(w_j) = 1         (convex combination)
#
#   These constraints are what separate Synthetic Control from regression.
#   They prevent extrapolation: the synthetic unit is a convex combination
#   of real donor states, so it stays within the convex hull of the data.
#   OLS would allow negative weights (which have no intuitive interpretation
#   as a 'weighted average of states') and can extrapolate outside the data.
#
# compute_rmspe:
#   Root Mean Squared Prediction Error on a given set of periods.
#   Pre-period RMSPE measures fit quality.
#   Post-period RMSPE measures treatment effect magnitude.
#   The ratio post/pre RMSPE is the inference statistic.
#
# compute_gap:
#   Gap = treated outcome - synthetic outcome, period by period.
#   Pre-period gaps should be near zero (good fit).
#   Post-period gaps are the estimated treatment effect.

def synthetic_control_weights(y_treated_pre, y_donors_pre):
    """
    Find optimal donor weights via constrained quadratic programming.
    Returns weight vector of length n_donors.
    """
    J = y_donors_pre.shape[1]
    
    # Scale. Without scaling the gradients explode and won't converge
    # Scale with a constant factor - here it is mean of y_treated_pre
    y_treated_pre, y_donors_pre = y_treated_pre / y_treated_pre.mean(), y_donors_pre/ y_treated_pre.mean()

    def objective(w):
        synth = y_donors_pre @ w
        return np.sum((y_treated_pre - synth) ** 2)

    def gradient(w):
        synth = y_donors_pre @ w
        return -2 * y_donors_pre.T @ (y_treated_pre - synth)

    # Initial guess: uniform weights
    w0 = np.ones(J) / J

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    # Bounds: each weight in [0, 1]
    bounds = [(0, 1)] * J

    result = minimize(objective, w0, jac=gradient,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-6, 'maxiter': 100000})

    if not result.success:
        print(f'  WARNING: optimizer did not converge: {result.message}')
    return result.x

def compute_synthetic(df, donor_cols, weights):
    """Compute synthetic control outcome for all periods."""
    return df[donor_cols].values @ weights

def compute_rmspe(actual, synthetic, mask):
    """Root mean squared prediction error on selected periods."""
    return np.sqrt(np.mean((actual[mask] - synthetic[mask]) ** 2))

# COMMAND ----------

# =============================================================================
# Fit Synthetic Control
# =============================================================================
#
# Step 1: Extract pre-treatment data for the treated state and all donors.
# Step 2: Solve the QP to get donor weights.
# Step 3: Construct the synthetic control outcome for all periods.
# Step 4: Compute the gap and ATT.
#
# INTERPRETING THE WEIGHTS:
#   Most weights will be zero -- the optimizer selects a sparse combination
#   of donor states that best match the treated state's pre-period trajectory.
#   A weight of 0.4 on donor_3 means 40% of the synthetic state's income
#   is drawn from donor_3's actual outcome. This is directly interpretable.
#
# PRE-PERIOD RMSPE:
#   The quality of the fit. If RMSPE is large relative to the scale of
#   the outcome, the synthetic control does not closely match the treated
#   state's pre-period trajectory -- and post-period gaps are unreliable.
#   Rule of thumb: RMSPE / mean(Y_treated_pre) < 2% is good.

y_treated    = df['treated'].values
y_donors_pre = df.loc[PRE_MASK, DONOR_COLS].values
y_treated_pre= df.loc[PRE_MASK, 'treated'].values

# Solve for weights
weights = synthetic_control_weights(y_treated_pre, y_donors_pre)

# Build synthetic control for all periods
y_synthetic  = compute_synthetic(df, DONOR_COLS, weights)
gap          = y_treated - y_synthetic

rmspe_pre    = compute_rmspe(y_treated, y_synthetic, PRE_MASK)
rmspe_post   = compute_rmspe(y_treated, y_synthetic, POST_MASK)
relative_rmspe_pre = rmspe_pre/y_treated_pre.mean()
att_estimate = gap[POST_MASK].mean()

# Donor weight summary
weight_df = pd.DataFrame({'donor': DONOR_COLS, 'weight': weights})
weight_df = weight_df[weight_df['weight'] > 0.001].sort_values('weight', ascending=False)

print("Donor weights (non-zero):")
print(weight_df.round(4).to_string(index=False))
print(f"\nPre-period RMSPE:   ${rmspe_pre:,.0f}")
print(f"Post-period RMSPE:  ${rmspe_post:,.0f}")
print(f"\nRelative RMSPE {100 * relative_rmspe_pre:,.2f}%")
if relative_rmspe_pre < .02:
    print('\u2713 Good pre-period fit')
print(f"\nPost/Pre ratio:     {rmspe_post/rmspe_pre:.2f}")
print(f"\nATT estimate:       ${att_estimate:,.0f}")
print(f"True ATT:           ${TRUE_EFFECT:,}")
print(f"Recovery:           {att_estimate/TRUE_EFFECT*100:.1f}%")


# COMMAND ----------

# =============================================================================
# Pre-Period Fit Visualization
# =============================================================================
#
# The most important diagnostic for Synthetic Control: how closely does the
# synthetic state track the treated state before the intervention?
#
# If pre-period fit is poor:
#   - The synthetic control is not a credible counterfactual
#   - Post-period gaps reflect pre-existing differences, not the treatment
#   - The method should be questioned or donor pool reconsidered
#
# If pre-period fit is good:
#   - The synthetic control is tracking the treated state on its own dynamics
#   - Post-period divergence is plausibly attributed to the treatment
#   - The ATT estimate is credible

periods = df.index.values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phase 5 -- Synthetic Control: Pre-Period Fit',
             fontsize=13, fontweight='bold')

# -- Left: full trajectory --------------------------------------------------
ax = axes[0]
ax.plot(periods, y_treated,   color='#e06b5b', lw=2.5,
        label='Treated state',     zorder=3)
ax.plot(periods, y_synthetic, color='#5b8dd9', lw=2.5, ls='--',
        label='Synthetic control', zorder=3)
ax.axvline(T_TREAT - 0.5, color='#333', ls='--', lw=1.2,
           label=f'Treatment (t={T_TREAT})')
ax.axvspan(T_TREAT - 0.5, T_TOTAL + 0.5, alpha=0.06, color='#e06b5b')
ax.set_xlabel('Period')
ax.set_ylabel('Per-capita income ($)')
ax.set_title('Treated vs Synthetic Control\n(full timeline)', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# -- Right: pre-period only, gap shaded ------------------------------------
ax = axes[1]
ax.plot(periods[PRE_MASK], y_treated[PRE_MASK],   color='#e06b5b', lw=2.5,
        label='Treated state')
ax.plot(periods[PRE_MASK], y_synthetic[PRE_MASK], color='#5b8dd9', lw=2.5, ls='--',
        label='Synthetic control')
ax.fill_between(periods[PRE_MASK], y_treated[PRE_MASK], y_synthetic[PRE_MASK],
                alpha=0.15, color='#e06b5b', label='Residual gap')
ax.set_xlabel('Period')
ax.set_ylabel('Per-capita income ($)')
rel_rmspe = rmspe_pre / y_treated[PRE_MASK].mean()
ax.set_title(f'Pre-period fit\nRMSPE = ${rmspe_pre:,.0f} ({rel_rmspe:.1%} of mean pre-treatment income)', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

plt.tight_layout()
plt.savefig('../assets/plots/phase5/prefit.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Pre-period fit plot saved")

# COMMAND ----------

# =============================================================================
# Gap Plot (Treated - Synthetic)
# =============================================================================
#
# The gap plot is the core result of Synthetic Control.
# Pre-period gap should hover near zero -- that's the definition of good fit.
# Post-period gap is the estimated treatment effect period by period.
#
# The ATT estimate is the average post-period gap.
# In the absence of treatment, we would expect the gap to remain near zero --
# the identifying assumption is that the synthetic control continues to track
# the treated state's counterfactual trajectory after treatment.

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle('Phase 5 -- Synthetic Control: Gap Plot (Treated - Synthetic)',
             fontsize=13, fontweight='bold')

ax.plot(periods, gap, color='#e06b5b', lw=2.5, label='Gap (treated - synthetic)')
ax.axhline(0, color='#333', lw=1, ls='-')
ax.axvline(T_TREAT - 0.5, color='#333', ls='--', lw=1.2,
           label=f'Treatment (t={T_TREAT})')
ax.axhline(att_estimate, color='#27ae60', ls='--', lw=1.5,
           label=f'ATT = ${att_estimate:,.0f}')
ax.axhline(TRUE_EFFECT,  color='#f39c12', ls=':', lw=1.5,
           label=f'True effect = ${TRUE_EFFECT:,}')
ax.fill_between(periods, gap, 0,
                where=POST_MASK, alpha=0.12, color='#27ae60',
                label='Post-period ATT region')
ax.set_xlabel('Period')
ax.set_ylabel('Gap ($)')
ax.set_title('Pre-period: gap near zero = good fit\n'
             'Post-period: gap = estimated treatment effect', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('../assets/plots/phase5/gap.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\u2713 Gap plot saved")
print(f"  Pre-period mean gap:  ${gap[PRE_MASK].mean():,.0f}  (should be ~0)")
print(f"  Post-period mean gap: ${gap[POST_MASK].mean():,.0f}  (ATT estimate)")


# COMMAND ----------

# =============================================================================
# Placebo-in-Time
# =============================================================================
#
# Idea: pretend treatment happened at an earlier date within the pre-period.
# If Synthetic Control is working correctly, a placebo treatment date should
# produce a near-zero 'effect' -- because nothing actually changed.
#
# If the placebo produces a large 'effect', it suggests the pre-period fit
# is not as clean as it looks -- the synthetic control is tracking noise,
# not the treated state's underlying dynamics.
#
# We test three placebo dates: t=10, t=13, t=16 (all in the pre-period).
# For each, we fit Synthetic Control using only periods before the placebo
# date for training, and measure the 'effect' in the remaining pre-period.
# A credible result shows these placebo effects are small.

PLACEBO_DATES = [10, 13, 16]

fig, axes = plt.subplots(1, len(PLACEBO_DATES), figsize=(15, 5), sharey=True)
fig.suptitle('Phase 5 -- Placebo-in-Time: No Effect Should Appear Before Treatment',
             fontsize=13, fontweight='bold')

print('Placebo-in-time results:')
print(f'{"Placebo date":<16} {"Placebo ATT":>14}  {"Interpretation"}')
print('\u2500' * 52)

for ax, t_placebo in zip(axes, PLACEBO_DATES):
    # Training: periods before placebo date
    train_mask   = df.index < t_placebo
    eval_mask    = (df.index >= t_placebo) & (df.index < T_TREAT)

    y_tr_pre  = df.loc[train_mask, 'treated'].values
    y_do_pre  = df.loc[train_mask, DONOR_COLS].values
    w_placebo = synthetic_control_weights(y_tr_pre, y_do_pre)

    y_synth_full = compute_synthetic(df, DONOR_COLS, w_placebo)
    gap_placebo  = y_treated - y_synth_full
    placebo_att  = gap_placebo[eval_mask].mean()

    status = '\u2713 small' if abs(placebo_att) < 300 else '\u26a0 large'
    print(f't = {t_placebo:<12}   ${placebo_att:>10,.0f}   {status}')

    ax.plot(df.index[~POST_MASK], gap_placebo[~POST_MASK],
            color='#5b8dd9', lw=2)
    ax.axvline(t_placebo - 0.5, color='#e06b5b', ls='--', lw=1.5,
               label=f'Placebo t={t_placebo}')
    ax.axvline(T_TREAT - 0.5, color='#333', ls=':', lw=1,
               label=f'True t={T_TREAT}')
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xlabel('Period')
    ax.set_title(f'Placebo t={t_placebo}\nPlacebo ATT = ${placebo_att:,.0f}',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

axes[0].set_ylabel('Gap ($)')
plt.tight_layout()
plt.savefig('../assets/plots/phase5/placebo_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Placebo-in-time plot saved")


# COMMAND ----------

# =============================================================================
# Placebo-in-Space (Permutation Inference)
# =============================================================================
#
# The primary inference method for Synthetic Control.
#
# Idea: apply the exact same Synthetic Control procedure to each donor state
# as if it were the treated state, using the remaining donors as the donor pool.
# Each donor gets a 'placebo gap' series.
#
# The key statistic is the post/pre RMSPE ratio:
#   ratio = RMSPE_post / RMSPE_pre
#
# A high ratio means the gap grew substantially post-treatment -- consistent
# with a real treatment effect. A low ratio means the gap is similar pre and
# post -- consistent with no effect.
#
# Inference: if the treated state's ratio is extreme relative to the placebo
# distribution, we have evidence that the effect is real.
# p-value = rank(treated ratio) / (n_placebos + 1)
#
# IMPORTANT: donors with very poor pre-period fit are excluded from the
# placebo distribution. If a donor has pre-period RMSPE >> treated state's
# pre-period RMSPE, it simply cannot be fit well -- including it inflates
# the placebo ratio distribution and makes the test conservative.
# Standard threshold: exclude placebos with pre-RMSPE > 2x treated pre-RMSPE.

RMSPE_THRESHOLD = 2.0   # exclude placebos with pre-RMSPE > threshold * treated

placebo_gaps   = {}   # donor -> gap array
placebo_ratios = {}   # donor -> post/pre RMSPE ratio
excluded       = []

for donor in DONOR_COLS:
    # Donor pool: all other donors. Exclude "treated"
    other_donors = [d for d in DONOR_COLS if d != donor]

    y_donor_pre  = df.loc[PRE_MASK, donor].values
    y_pool_pre   = df.loc[PRE_MASK, other_donors].values

    w_pl    = synthetic_control_weights(y_donor_pre, y_pool_pre)
    y_synth = compute_synthetic(df, other_donors, w_pl)
    y_donor = df[donor].values
    g       = y_donor - y_synth

    rmspe_pre_pl  = compute_rmspe(y_donor, y_synth, PRE_MASK)
    rmspe_post_pl = compute_rmspe(y_donor, y_synth, POST_MASK)

    # Exclude if pre-period fit is much worse than treated state
    if rmspe_pre_pl > RMSPE_THRESHOLD * rmspe_pre:
        excluded.append(donor)
        continue

    ratio = rmspe_post_pl / rmspe_pre_pl if rmspe_pre_pl > 0 else np.inf
    placebo_gaps[donor]   = g
    placebo_ratios[donor] = ratio

# Treated state ratio
treated_ratio = rmspe_post / rmspe_pre

# p-value: fraction of placebos with ratio >= treated ratio
all_ratios    = list(placebo_ratios.values())
p_value       = np.mean([r >= treated_ratio for r in all_ratios])

print(f"Treated state  post/pre RMSPE ratio: {treated_ratio:.2f}")
print(f"Placebo ratios (mean):               {np.mean(all_ratios):.2f}")
print(f"Placebo ratios (max):                {np.max(all_ratios):.2f}")
print(f"Excluded (poor pre-fit):             {len(excluded)} donors")
print(f"Valid placebos:                      {len(all_ratios)}")
print(f"\np-value (permutation):              {p_value:.3f}")
print(f"  (fraction of placebos with ratio >= {treated_ratio:.2f})")

# COMMAND ----------

# =============================================================================
# Placebo-in-Space Visualization
# =============================================================================
#
# Two panels:
#   Left:  All gap series (treated + placebos). The treated state's gap should
#          stand out post-treatment. Pre-period gaps should be similar.
#   Right: Distribution of post/pre RMSPE ratios. The treated state's ratio
#          should be in the right tail.

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phase 5 -- Placebo-in-Space: Permutation Inference',
             fontsize=13, fontweight='bold')

# -- Left: gap series -------------------------------------------------------
ax = axes[0]
for donor, g in placebo_gaps.items():
    ax.plot(periods, g, color='#aaaaaa', lw=0.8, alpha=0.5, zorder=1)
ax.plot(periods, gap, color='#e06b5b', lw=2.5,
        label=f'Treated state (ratio={treated_ratio:.1f})', zorder=3)
ax.axvline(T_TREAT - 0.5, color='#333', ls='--', lw=1.2,
           label=f'Treatment (t={T_TREAT})')
ax.axhline(0, color='#333', lw=0.8)
ax.set_xlabel('Period')
ax.set_ylabel('Gap ($)')
ax.set_title('Gap series: treated (red) vs placebos (grey)\n'
             'Treated gap should stand out post-treatment', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# -- Right: RMSPE ratio distribution ----------------------------------------
ax = axes[1]
ax.hist(all_ratios, bins=15, color='#5b8dd9', alpha=0.75,
        label=f'Placebo ratios (n={len(all_ratios)})', edgecolor='white')
ax.axvline(treated_ratio, color='#e06b5b', lw=2.5,
           label=f'Treated state ({treated_ratio:.1f})')
ax.set_xlabel('Post/Pre RMSPE ratio')
ax.set_ylabel('Count')
ax.set_title(f'RMSPE ratio distribution\np-value = {p_value:.3f}',
             fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../assets/plots/phase5/placebo_space.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Placebo-in-space plot saved")

# COMMAND ----------

# =============================================================================
# Full Results Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle('Phase 5 -- Synthetic Control Results',
             fontsize=13, fontweight='bold')

# -- Plot A: Treated vs Synthetic (full timeline) ---------------------------
ax = fig.add_subplot(gs[0, 0])
ax.plot(periods, y_treated,   color='#e06b5b', lw=2.5, label='Treated state')
ax.plot(periods, y_synthetic, color='#5b8dd9', lw=2.5, ls='--',
        label='Synthetic control')
ax.axvline(T_TREAT - 0.5, color='#333', ls='--', lw=1.2,
           label=f'Treatment t={T_TREAT}')
ax.axvspan(T_TREAT - 0.5, T_TOTAL + 0.5, alpha=0.06, color='#e06b5b')
ax.set_xlabel('Period')
ax.set_ylabel('Per-capita income ($)')
ax.set_title('Treated vs Synthetic Control', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# -- Plot B: Gap plot -------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.plot(periods, gap, color='#e06b5b', lw=2.5, label='Gap')
ax.axvline(T_TREAT - 0.5, color='#333', ls='--', lw=1.2)
ax.axhline(0, color='#333', lw=0.8)
ax.axhline(att_estimate, color='#27ae60', ls='--', lw=1.5,
           label=f'ATT = ${att_estimate:,.0f}')
ax.axhline(TRUE_EFFECT, color='#f39c12', ls=':', lw=1.5,
           label=f'True = ${TRUE_EFFECT:,}')
ax.fill_between(periods, gap, 0, where=POST_MASK,
                alpha=0.12, color='#27ae60')
ax.set_xlabel('Period')
ax.set_ylabel('Gap ($)')
ax.set_title('Gap: Treated - Synthetic', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# -- Plot C: Donor weights bar chart ----------------------------------------
ax = fig.add_subplot(gs[0, 2])
w_nonzero = weight_df.sort_values('weight').tail(10)  # top 10 donors
ax.barh(w_nonzero['donor'], w_nonzero['weight'],
        color='#5b8dd9', alpha=0.85)
ax.set_xlabel('Weight')
ax.set_title('Donor weights\n(non-zero, top 10)', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.axvline(0, color='#333', lw=0.8)

# -- Plot D: Placebo-in-space gap series ------------------------------------
ax = fig.add_subplot(gs[1, 0])
for donor, g in placebo_gaps.items():
    ax.plot(periods, g, color='#aaaaaa', lw=0.8, alpha=0.5)
ax.plot(periods, gap, color='#e06b5b', lw=2.5,
        label='Treated state')
ax.axvline(T_TREAT - 0.5, color='#333', ls='--', lw=1.2)
ax.axhline(0, color='#333', lw=0.8)
ax.set_xlabel('Period')
ax.set_ylabel('Gap ($)')
ax.set_title('Placebo-in-space\ngrey = donor placebos', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# -- Plot E: RMSPE ratio distribution ---------------------------------------
ax = fig.add_subplot(gs[1, 1])
ax.hist(all_ratios, bins=15, color='#5b8dd9', alpha=0.75,
        edgecolor='white', label='Placebo ratios')
ax.axvline(treated_ratio, color='#e06b5b', lw=2.5,
           label=f'Treated ({treated_ratio:.1f})')
ax.set_xlabel('Post/Pre RMSPE ratio')
ax.set_ylabel('Count')
ax.set_title(f'Permutation inference\np = {p_value:.3f}', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Plot F: ATT estimate over post-period ----------------------------------
ax = fig.add_subplot(gs[1, 2])
post_periods = periods[POST_MASK]
post_gap     = gap[POST_MASK]
ax.bar(post_periods, post_gap, color='#27ae60', alpha=0.8, width=0.7,
       label='Period ATT')
ax.axhline(att_estimate, color='#27ae60', ls='--', lw=1.5,
           label=f'Mean ATT = ${att_estimate:,.0f}')
ax.axhline(TRUE_EFFECT, color='#f39c12', ls=':', lw=1.5,
           label=f'True = ${TRUE_EFFECT:,}')
ax.axhline(0, color='#333', lw=0.8)
ax.set_xlabel('Period')
ax.set_ylabel('Estimated effect ($)')
ax.set_title('Period-by-period ATT\npost-treatment', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.savefig('../assets/plots/phase5/results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Results plot saved")


# COMMAND ----------

# =============================================================================
# MLflow Logging
# =============================================================================

mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/causal_inference_toolkit")

with mlflow.start_run(run_name='phase5_synthetic_control'):

    # Parameters
    mlflow.log_param('method',            'Synthetic Control')
    mlflow.log_param('dataset',           'Simulated state panel')
    mlflow.log_param('estimand',          'ATT')
    mlflow.log_param('n_donors',          N_DONORS)
    mlflow.log_param('t_total',           T_TOTAL)
    mlflow.log_param('t_treat',           T_TREAT)
    mlflow.log_param('pre_periods',       T_TREAT - 1)
    mlflow.log_param('post_periods',      T_TOTAL - T_TREAT + 1)
    mlflow.log_param('optimizer',         'SLSQP')
    mlflow.log_param('rmspe_threshold',   RMSPE_THRESHOLD)
    mlflow.log_param('true_effect',       TRUE_EFFECT)

    # Metrics
    mlflow.log_metric('att_estimate',     round(att_estimate,   2))
    mlflow.log_metric('true_effect',      TRUE_EFFECT)
    mlflow.log_metric('recovery_pct',     round(att_estimate / TRUE_EFFECT * 100, 1))
    mlflow.log_metric('rmspe_pre',        round(rmspe_pre,      2))
    mlflow.log_metric('rmspe_post',       round(rmspe_post,     2))
    mlflow.log_metric('rmspe_ratio',      round(treated_ratio,  3))
    mlflow.log_metric('p_value',          round(p_value,        3))
    mlflow.log_metric('n_valid_placebos', len(all_ratios))
    mlflow.log_metric('n_excluded',       len(excluded))
    mlflow.log_metric('n_nonzero_donors', int((weights > 0.001).sum()))

    # Artifacts
    for fname in ['prefit', 'gap', 'placebo_time', 'placebo_space', 'results']:
        try:
            mlflow.log_artifact(f'../assets/plots/phase5/{fname}.png')
        except Exception:
            pass

    run_id = mlflow.active_run().info.run_id
    print(f"\n\u2713 MLflow run logged -- Run ID: {run_id}")
    print(f"  ATT estimate:    ${att_estimate:,.0f}")
    print(f"  True effect:     ${TRUE_EFFECT:,}")
    print(f"  Recovery:        {att_estimate/TRUE_EFFECT*100:.1f}%")
    print(f"  Pre RMSPE:       ${rmspe_pre:,.0f}")
    print(f"  RMSPE ratio:     {treated_ratio:.2f}")
    print(f"  p-value:         {p_value:.3f}")


# COMMAND ----------

# =============================================================================
# Summary and Bridge to Phase 6 (Regression Discontinuity)
# =============================================================================

print(f"""
What we did:
  \u2713 Simulated a state-level panel with 1 treated state, {N_DONORS} donors, T={T_TOTAL}
  \u2713 Embedded true ATT = ${TRUE_EFFECT:,} post t={T_TREAT}
  \u2713 Solved constrained QP to find optimal donor weights
  \u2713 Built synthetic control -- convex combination of donor outcomes
  \u2713 Computed pre-period RMSPE (fit quality) and post-period gap (ATT)
  \u2713 Placebo-in-time: tested spurious effects at pre-period dates
  \u2713 Placebo-in-space: applied method to each donor, built permutation test
  \u2713 Logged all results to MLflow

Key results:
  True ATT (embedded):    ${TRUE_EFFECT:,}
  ATT estimate:           ${att_estimate:,.0f}   ({att_estimate/TRUE_EFFECT*100:.1f}% recovery)
  Pre-period RMSPE:       ${rmspe_pre:,.0f}   (fit quality -- lower is better)
  Post/Pre RMSPE ratio:   {treated_ratio:.2f}
  Permutation p-value:    {p_value:.3f}
  Valid placebos:         {len(all_ratios)} of {N_DONORS} donors

What Synthetic Control does that IPW and PSM cannot:
  - Works when N_treated = 1. You cannot run IPW or PSM with one treated unit.
  - Uses the pre-treatment time series as the basis for the counterfactual.
    The counterfactual is grounded in actual historical co-movement, not
    cross-sectional covariate similarity.
  - Provides visual, interpretable inference: the gap plot shows exactly
    when the treated and synthetic trajectories diverge.
  - Inference via permutation (placebo-in-space) requires no distributional
    assumptions -- valid regardless of the error structure.

What Synthetic Control cannot do:
  - Handle many treated units (it was designed for N=1)
  - Work with short pre-periods (needs enough time to establish good fit)
  - Extrapolate outside the convex hull of donors (by design)
  - Estimate heterogeneous treatment effects across units

-> Phase 6 (Regression Discontinuity) works on a completely different
   identification strategy. Instead of a pre-treatment parallel trajectory
   or propensity score overlap, it exploits a discontinuity in a running
   variable: units just above and just below a threshold are assumed
   comparable except for treatment assignment. We simulate a scholarship
   cutoff dataset to demonstrate it.
""")

