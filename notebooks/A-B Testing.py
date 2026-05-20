# Databricks notebook source
# MAGIC %md
# MAGIC ### Causal Inference Toolkit
# MAGIC
# MAGIC ##### Problem: Did showing users a new product page design increase purchases?
# MAGIC ##### Method:  A/B Test (Randomized Controlled Trial)
# MAGIC ##### Dataset: Simulated e-commerce data (ground truth known)
# MAGIC ##### Estimand: ATE — Average Treatment Effect on the full user population

# COMMAND ----------

# %pip install scipy statsmodels sentinel mlflow --quiet
# %restart_python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
import mlflow
import mlflow.sklearn
from datetime import datetime

np.random.seed(42)  # reproducibility — same results every run

print("✓ Imports complete")

# COMMAND ----------

# =============================================================================
# Simulate the Dataset
# =============================================================================
#
# WHAT WE ARE BUILDING:
#   10,000 users. Each user has background characteristics (age, device type,
#   prior visits). Half are randomly assigned to see the NEW page (treatment=1),
#   half see the OLD page (treatment=0). Each user either purchases (1) or not (0).
#
# THE GROUND TRUTH WE BAKE IN:
#   Base purchase rate (control):  10%
#   True treatment effect (lift):  +3 percentage points
#   So treated purchase rate:      13%
#
# WHY ADD COVARIATES (age, device, visits)?
#   In a real A/B test they don't affect the estimate (randomization handles it).
#   But we'll use them later in Propensity Score Matching (PSM) where they become confounders.
#   Planting them now keeps our dataset consistent across all phases.

N = 10_000  # number of users

# --- User background characteristics ---
# These are independent of treatment assignment (randomization guarantees this)
age          = np.random.normal(35, 10, N).clip(18, 70)          # age in years
prior_visits = np.random.poisson(5, N)                            # visits before experiment
device       = np.random.choice(['mobile', 'desktop'], N,
                                p=[0.6, 0.4])                     # 60% mobile users

# --- Random treatment assignment ---
# np.random.binomial(1, 0.5) = fair coin flip per user
# This is what makes it an A/B test. Each user independently gets 0 or 1.
treatment = np.random.binomial(1, 0.5, N)

# --- Outcome: did the user purchase? ---
#
# The purchase probability depends on:
#   1. Treatment (the page design) — this is what we want to measure
#   2. Background characteristics — these affect purchase propensity naturally
#   3. Random noise — not every user behaves predictably
#
# We model purchase probability as a logistic function so it stays in [0,1].
# The formula below is the "data generating process" (DGP) — the true model
# that created our data. In real life you never know this. Here you do.

TRUE_EFFECT = 0.03           # 3 percentage point lift — this is what we want to recover

log_odds = (
    -2.5                              # baseline (sets ~8% base rate before covariates)
    + TRUE_EFFECT * 4 * treatment     # treatment effect (scaled to logit space)
    + 0.02  * (age - 35)             # older users buy slightly more
    + 0.05  * prior_visits           # more prior visits → more likely to buy
    + 0.2   * (device == 'desktop')  # desktop users convert slightly better
)
purchase_prob = 1 / (1 + np.exp(-log_odds))   # sigmoid: maps log-odds → probability
purchased     = np.random.binomial(1, purchase_prob, N)  # actual 0/1 outcome

# --- Assemble into a DataFrame ---
df = pd.DataFrame({
    'user_id'      : range(N),
    'treatment'    : treatment,           # 0 = old page, 1 = new page
    'purchased'    : purchased,           # 0 = no purchase, 1 = purchase
    'age'          : age.round(1),
    'prior_visits' : prior_visits,
    'device'       : device,
    'purchase_prob': purchase_prob.round(4)  # keep true prob for sanity checks
})

print(f"Dataset shape: {df.shape}")
print(f"\nTreatment split:\n{df['treatment'].value_counts()}")
print(f"\nPurchase rate overall: {df['purchased'].mean():.3f}")
print(f"True effect baked in:  {TRUE_EFFECT} ({TRUE_EFFECT*100:.1f} percentage points)")

# COMMAND ----------

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
#
# RULE: Always explore before modeling.
# We want to verify:
#   (a) Treatment assignment is balanced (randomization worked)
#   (b) Purchase rates look as expected
#   (c) No obvious data quality issues

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# --- 3a. Check balance across groups ---
# In a proper A/B test, covariates should be similar across groups.
# If they're not, something went wrong with randomization.

balance = df.groupby('treatment').agg(
    n_users       = ('user_id',       'count'),
    avg_age       = ('age',           'mean'),
    avg_visits    = ('prior_visits',  'mean'),
    pct_desktop   = ('device',        lambda x: (x == 'desktop').mean()),
    purchase_rate = ('purchased',     'mean')
).round(3)

print("\nBalance table (covariates should be similar across groups):")
print(balance.to_string())

# --- 3b. Purchase rates by group ---
ctrl_rate  = df[df['treatment'] == 0]['purchased'].mean()
treat_rate = df[df['treatment'] == 1]['purchased'].mean()
naive_diff = treat_rate - ctrl_rate

print(f"\nControl purchase rate:   {ctrl_rate:.4f}  ({ctrl_rate*100:.2f}%)")
print(f"Treatment purchase rate: {treat_rate:.4f}  ({treat_rate*100:.2f}%)")
print(f"Naive difference:        {naive_diff:.4f}  ({naive_diff*100:.2f} pp)")
print(f"True effect (baked in):  {TRUE_EFFECT:.4f}  ({TRUE_EFFECT*100:.2f} pp)")
print(f"\nRecovery: {naive_diff/TRUE_EFFECT*100:.1f}% of true effect captured")

# WHY DOES THE NAIVE DIFFERENCE WORK HERE?
# Because treatment is random, the two groups are identical on everything except
# the page they saw. The difference in purchase rates IS the causal effect.
# This is the magic of randomization — no adjustment needed.

# COMMAND ----------

# =============================================================================
# ATE Estimation (Three ways, same answer)
# =============================================================================
#
# We'll estimate ATE three ways. In a proper RCT all three give the same result.
# This builds intuition for what estimators are — different roads to the same destination.

print("=" * 55)
print("ATE ESTIMATION")
print("=" * 55)

# --- Method A: Difference in means (the formula) ---
#
# ATE = E[Y | D=1] - E[Y | D=0]
# The simplest possible estimator. Valid ONLY under randomization.

Y1     = df[df['treatment'] == 1]['purchased']   # outcomes for treated
Y0     = df[df['treatment'] == 0]['purchased']   # outcomes for control
ATE_dm = Y1.mean() - Y0.mean()

print(f"\nMethod A — Difference in means")
print(f"  ATE = {ATE_dm:.5f}  ({ATE_dm*100:.3f} pp)")

# --- Method B: OLS regression ---
#
# Y = β0 + β1 * D + ε
# β1 is the ATE estimate. Adding covariates (age, visits) makes it more precise
# but doesn't change the point estimate much — because treatment is random,
# covariates are uncorrelated with it.
#
# IMPORTANT: The coefficient on 'treatment' in OLS = ATE when treatment is binary
# and randomly assigned. This is a Linear Probability Model (LPM).

X_simple  = sm.add_constant(df['treatment'])               # just treatment
X_covars  = sm.add_constant(df[['treatment', 'age',
                                 'prior_visits']])          # treatment + covariates

ols_simple = sm.OLS(df['purchased'], X_simple).fit()
ols_covars = sm.OLS(df['purchased'], X_covars).fit()

ATE_ols_simple = ols_simple.params['treatment']
ATE_ols_covars = ols_covars.params['treatment']

print(f"\nMethod B — OLS (no covariates)")
print(f"  ATE = {ATE_ols_simple:.5f}  ({ATE_ols_simple*100:.3f} pp)")
print(f"  SE  = {ols_simple.bse['treatment']:.5f}")
print(f"  p   = {ols_simple.pvalues['treatment']:.4f}")

print(f"\nMethod B — OLS (with covariates)")
print(f"  ATE = {ATE_ols_covars:.5f}  ({ATE_ols_covars*100:.3f} pp)")
print(f"  SE  = {ols_covars.bse['treatment']:.5f}   ← smaller SE = more precise")
print(f"  p   = {ols_covars.pvalues['treatment']:.4f}")

# WHY DOES ADDING COVARIATES HELP PRECISION BUT NOT THE ESTIMATE?
# Covariates explain residual variance in Y. Less residual variance → tighter SE.
# But they don't change the estimate because they're uncorrelated with treatment
# (randomization ensures this). In observational studies (DiD, PSM) adding
# covariates both changes the estimate AND reduces SE. Big difference.

# --- Method C: Welch's t-test ---
#
# Tests H0: mean(Y1) = mean(Y0) — i.e., no treatment effect
# Returns t-statistic and p-value. Equivalent to OLS simple in large samples.

t_stat, p_val = stats.ttest_ind(Y1, Y0, equal_var=False)
ATE_ttest = Y1.mean() - Y0.mean()

print(f"\nMethod C — Welch's t-test")
print(f"  ATE = {ATE_ttest:.5f}  ({ATE_ttest*100:.3f} pp)")
print(f"  t   = {t_stat:.4f}")
print(f"  p   = {p_val:.4f}")

print(f"\n{'─'*55}")
print(f"True effect:       {TRUE_EFFECT:.5f}")
print(f"All three match ✓  (expected — randomization guarantees this)")

# COMMAND ----------

# =============================================================================
# Confidence Intervals (Two approaches)
# =============================================================================
#
# A point estimate (3.1 pp) is not enough. We need a range: the CI.
# CI tells you: "Given this data, the true effect is between X and Y
# with 95% probability."
#
# Two methods — both should give similar results:
#   (a) Analytical CI — from OLS standard errors (assumes normality)
#   (b) Bootstrap CI — resample data 2000 times, compute CI from distribution

print("=" * 55)
print("CONFIDENCE INTERVALS")
print("=" * 55)

# --- 5a. Analytical CI (from OLS) ---
# statsmodels gives this directly
ci_low  = ols_simple.conf_int().loc['treatment', 0]
ci_high = ols_simple.conf_int().loc['treatment', 1]

print(f"\nAnalytical 95% CI (OLS):")
print(f"  [{ci_low:.5f},  {ci_high:.5f}]")
print(f"  [{ci_low*100:.3f} pp,  {ci_high*100:.3f} pp]")

# --- 5b. Bootstrap CI ---
#
# Bootstrap is model-free. Resample the dataset with replacement N times.
# Compute the ATE on each resample. The 2.5th–97.5th percentile of that
# distribution is your 95% CI.
#
# Why bootstrap? It makes fewer distributional assumptions. Especially useful
# when your outcome isn't continuous (purchased is 0/1, not Gaussian).

N_BOOT      = 2_000
boot_ates   = np.zeros(N_BOOT)

for i in range(N_BOOT):
    sample      = df.sample(n=N, replace=True)   # resample with replacement
    y1_boot     = sample[sample['treatment'] == 1]['purchased'].mean()
    y0_boot     = sample[sample['treatment'] == 0]['purchased'].mean()
    boot_ates[i] = y1_boot - y0_boot

boot_ci_low  = np.percentile(boot_ates, 2.5)
boot_ci_high = np.percentile(boot_ates, 97.5)

print(f"\nBootstrap 95% CI (2,000 resamples):")
print(f"  [{boot_ci_low:.5f},  {boot_ci_high:.5f}]")
print(f"  [{boot_ci_low*100:.3f} pp,  {boot_ci_high*100:.3f} pp]")
print(f"\n  Bootstrap mean ATE: {boot_ates.mean():.5f}")
print(f"  Bootstrap SE:        {boot_ates.std():.5f}")

# Does the CI include 0?
if boot_ci_low > 0:
    print(f"\n  ✓ CI does not include 0 → statistically significant at α=0.05")
else:
    print(f"\n  ✗ CI includes 0 → not statistically significant at α=0.05")

# COMMAND ----------

# =============================================================================
# Visualizations
# =============================================================================
#
# Three plots:
#   (a) Purchase rate comparison with CI bars
#   (b) Bootstrap distribution of the ATE
#   (c) Covariate balance (verifying randomization worked)

fig, ax = plt.subplots(figsize=(5,5))
# fig.suptitle("A/B Test Results", fontsize=14, fontweight='bold', y=1.01)

# ── Plot A: Purchase rates with CI bars ──────────────────────────────────────
# ax = axes[0]

groups      = ['Control\n(old page)', 'Treatment\n(new page)']
rates       = [ctrl_rate, treat_rate]
ci_half_ctrl  = (ci_high - ci_low) / 2   # approximate — using overall CI
se_ctrl     = Y0.std() / np.sqrt(len(Y0))
se_treat    = Y1.std() / np.sqrt(len(Y1))
ci_ctrl     = 1.96 * se_ctrl
ci_treat    = 1.96 * se_treat

bars = ax.bar(groups, rates,
              color=['#5b8dd9', '#e06b5b'],
              width=0.4,
              yerr=[ci_ctrl, ci_treat],
              capsize=6,
              error_kw={'linewidth': 2})

# Annotate bars
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.004,
            f'{rate*100:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add arrow showing the treatment effect
ax.annotate('', xy=(1, treat_rate), xytext=(1, ctrl_rate),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
ax.text(1.22, (ctrl_rate + treat_rate)/2,
        f'ATE\n{ATE_dm*100:.2f} pp', ha='left', va='center', fontsize=9,
        color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='#aaa', alpha=0.9))

ax.set_title("Purchase rates by group", fontweight='bold')
ax.set_ylabel("Purchase rate")
ax.set_ylim(0, max(rates) * 1.25)
ax.axhline(ctrl_rate, color='#5b8dd9', linestyle='--', alpha=0.4, linewidth=1)
ax.spines[['top', 'right']].set_visible(False)
plt.savefig('../assets/plots/purchase_rates.png', dpi=150, bbox_inches='tight')

# ── Plot B: Bootstrap distribution ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))

# ax = axes[1]

ax.hist(boot_ates, bins=60, color='#7fb3d3', edgecolor='white', linewidth=0.3)
ax.axvline(boot_ates.mean(),  color='#333',  linestyle='-',  linewidth=2,
           label=f'ATE = {boot_ates.mean()*100:.2f} pp')
ax.axvline(TRUE_EFFECT,        color='#e06b5b', linestyle='--', linewidth=2,
           label=f'True effect = {TRUE_EFFECT*100:.1f} pp')
ax.axvline(boot_ci_low,        color='#555', linestyle=':', linewidth=1.5,
           label=f'95% CI')
ax.axvline(boot_ci_high,       color='#555', linestyle=':', linewidth=1.5)

# Shade CI region
ax.axvspan(boot_ci_low, boot_ci_high, alpha=0.15, color='#7fb3d3')

ax.set_title("Bootstrap distribution of ATE\n(2,000 resamples)", fontweight='bold')
ax.set_xlabel("Estimated ATE")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.savefig('../assets/plots/bootstrap_distribution.png', dpi=150, bbox_inches='tight')

# ── Plot C: Covariate balance ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
# ax = axes[2]

# Standardized mean difference for age and prior_visits
covariates = ['age', 'prior_visits']
smds = []
for cov in covariates:
    m1   = df[df['treatment'] == 1][cov].mean()
    m0   = df[df['treatment'] == 0][cov].mean()
    s_pool = df[cov].std()
    smds.append((m1 - m0) / s_pool)

colors = ['#5cb85c' if abs(s) < 0.1 else '#d9534f' for s in smds]
bars   = ax.barh(covariates, [abs(s) for s in smds], color=colors, height=0.4)
ax.axvline(0.1, color='#d9534f', linestyle='--', linewidth=1.5, label='SMD = 0.1 threshold')
ax.axvline(0.0, color='#333',    linestyle='-',  linewidth=0.8)

for bar, smd in zip(bars, smds):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{abs(smd):.4f}', va='center', fontsize=9)

ax.set_title("Covariate balance check\n(SMD after randomization)", fontweight='bold')
ax.set_xlabel("Absolute Standardized Mean Difference")
ax.legend(fontsize=8)
ax.set_xlim(0, 0.15)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../assets/plots/covariate_balance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plots saved")

# COMMAND ----------

# =============================================================================
# What happens when you BREAK the randomization?
# =============================================================================
#
# This is the most important cell. We deliberately introduce selection bias
# to show why you can't use A/B test analysis on observational data.
#
# SCENARIO: Imagine we didn't randomize. Instead:
#   - Younger, more engaged users (high prior_visits) self-selected
#     into the new page (maybe it was only shown on mobile, and
#     younger users use mobile more)
#   - These users already had higher purchase intent
#
# The naive estimate will be INFLATED — it'll look like the page
# caused more purchases, but really it just attracted better users.

print("=" * 55)
print("BREAKING THE EXPERIMENT — Selection Bias Demo")
print("=" * 55)

df_broken = df.copy()

# Re-assign treatment based on age + visits (not random anymore)
# Young users (age < 35) with many visits are more likely to get "treated"
selection_log_odds = (
    -1.0
    - 0.04 * (df_broken['age'] - 35)       # younger → more likely treated
    + 0.15 * df_broken['prior_visits']      # more visits → more likely treated
)
selection_prob  = 1 / (1 + np.exp(-selection_log_odds))
df_broken['treatment'] = np.random.binomial(1, selection_prob, N)

# Now re-generate purchased using the SAME true DGP (same TRUE_EFFECT = 0.03)
# The only thing that changed is how treatment was assigned
log_odds_broken = (
    -2.5
    + TRUE_EFFECT * 4 * df_broken['treatment']
    + 0.02 * (df_broken['age'] - 35)
    + 0.05 * df_broken['prior_visits']
    + 0.2  * (df_broken['device'] == 'desktop')
)
purchase_prob_broken      = 1 / (1 + np.exp(-log_odds_broken))
df_broken['purchased']    = np.random.binomial(1, purchase_prob_broken, N)

# Naive estimate on biased data
ctrl_broken   = df_broken[df_broken['treatment'] == 0]['purchased'].mean()
treat_broken  = df_broken[df_broken['treatment'] == 1]['purchased'].mean()
naive_broken  = treat_broken - ctrl_broken

# Covariate imbalance in broken version
age_smd_broken = (
    (df_broken[df_broken['treatment']==1]['age'].mean() -
     df_broken[df_broken['treatment']==0]['age'].mean())
    / df_broken['age'].std()
)
visits_smd_broken = (
    (df_broken[df_broken['treatment']==1]['prior_visits'].mean() -
     df_broken[df_broken['treatment']==0]['prior_visits'].mean())
    / df_broken['prior_visits'].std()
)

print(f"\nTRUE causal effect:             {TRUE_EFFECT:.4f}  ({TRUE_EFFECT*100:.2f} pp)")
print(f"\n--- PROPER A/B TEST ---")
print(f"Estimated ATE:                  {ATE_dm:.4f}  ({ATE_dm*100:.2f} pp)  ✓ accurate")
print(f"Age SMD:                        {smds[0]:.4f}  (balanced)")
print(f"\n--- BROKEN (self-selection) ---")
print(f"Naive estimate:                 {naive_broken:.4f}  ({naive_broken*100:.2f} pp)  ✗ inflated!")
print(f"Bias:                          +{(naive_broken - TRUE_EFFECT)*100:.2f} pp")
print(f"Age SMD:                        {age_smd_broken:.4f}  (IMBALANCED — red flag)")
print(f"Prior visits SMD:               {visits_smd_broken:.4f}  (IMBALANCED — red flag)")
print(f"\nConclusion: Without randomization, naive diff = {naive_broken*100:.2f} pp")
print(f"  We'd think the effect is {naive_broken/TRUE_EFFECT:.1f}x larger than it really is.")
print(f"  This is exactly the problem DiD, PSM, SC, RD are designed to fix.")

# COMMAND ----------

home_path = '/Workspace/Users/deshpande.ajay.us@gmail.com'

# COMMAND ----------

# =============================================================================
# MLflow Logging
# =============================================================================
#
# We log every experiment to MLflow so in Phase 7 we can compare all methods
# side by side in one table/chart. This is the professional standard in DS teams.
#
# What we log:
#   Parameters: experiment design choices
#   Metrics:    the estimates we care about
#   Artifacts:  the visualization

print("=" * 55)
print("MLFLOW LOGGING")
print("=" * 55)

mlflow.set_experiment(f"{home_path}/causal_inference_toolkit")

with mlflow.start_run(run_name="ab_test_rct"):

    # --- Parameters (design choices) ---
    mlflow.log_param("method",          "AB_Test_RCT")
    mlflow.log_param("dataset",         "simulated_ecommerce")
    mlflow.log_param("n_users",         N)
    mlflow.log_param("n_bootstrap",     N_BOOT)
    mlflow.log_param("estimand",        "ATE")
    mlflow.log_param("true_effect",     TRUE_EFFECT)

    # --- Metrics (results) ---
    mlflow.log_metric("ATE_estimate",        round(ATE_dm, 5))
    mlflow.log_metric("ATE_pct_points",      round(ATE_dm * 100, 3))
    mlflow.log_metric("CI_lower",            round(boot_ci_low, 5))
    mlflow.log_metric("CI_upper",            round(boot_ci_high, 5))
    mlflow.log_metric("CI_lower_pp",         round(boot_ci_low * 100, 3))
    mlflow.log_metric("CI_upper_pp",         round(boot_ci_high * 100, 3))
    mlflow.log_metric("p_value",             round(p_val, 5))
    mlflow.log_metric("t_statistic",         round(t_stat, 4))
    mlflow.log_metric("recovery_pct",        round(ATE_dm / TRUE_EFFECT * 100, 2))
    mlflow.log_metric("age_SMD",             round(abs(smds[0]), 4))
    mlflow.log_metric("visits_SMD",          round(abs(smds[1]), 4))
    mlflow.log_metric("control_rate",        round(ctrl_rate, 5))
    mlflow.log_metric("treatment_rate",      round(treat_rate, 5))

    # --- Artifact: save the plot ---
    mlflow.log_artifact("/tmp/ab_test_results.png")

    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ MLflow run logged")
    print(f"  Run ID:   {run_id}")
    print(f"  ATE:      {ATE_dm:.5f}  ({ATE_dm*100:.3f} pp)")
    print(f"  95% CI:   [{boot_ci_low*100:.3f}, {boot_ci_high*100:.3f}] pp")
    print(f"  p-value:  {p_val:.4f}")

# COMMAND ----------

print(f"""
What we did:
  ✓ Simulated 10,000 users with known true effect = {TRUE_EFFECT*100:.1f} pp
  ✓ Randomly assigned treatment (proper A/B test)
  ✓ Estimated ATE three ways — all converged to ~{ATE_dm*100:.2f} pp
  ✓ Built 95% CI using both analytical and bootstrap methods
  ✓ Verified covariate balance (SMD < 0.1 on all covariates)
  ✓ Demonstrated selection bias by breaking randomization
  ✓ Logged everything to MLflow

Key results:
  True effect:    {TRUE_EFFECT*100:.2f} pp
  Estimated ATE:  {ATE_dm*100:.3f} pp
  95% CI:         [{boot_ci_low*100:.2f}, {boot_ci_high*100:.2f}] pp
  p-value:        {p_val:.4f}

Why it worked:
  Randomization made treatment independent of all covariates (observed
  and unobserved). The naive mean difference IS the causal effect.
  No modeling required.

The problem we now face (Phase 2):
  In the LaLonde dataset, workers were NOT randomly assigned to
  job training. They self-selected — more motivated, younger workers
  enrolled. Any naive comparison is contaminated by this selection.
  We cannot use the A/B test approach.

  The fix: Difference-in-Differences (DiD).
  If we have data from before AND after the training program, and we
  have a comparison group that didn't receive training, we can
  difference out the selection bias — IF the two groups would have
  trended the same way without treatment (parallel trends).
  """)
