# Databricks notebook source
# Phase 3 — Propensity Score Matching (PSM)
# Applied Causal Inference Series
#
# Problem:  Does job training increase earnings? Match similar workers and compare.
# Method:   Propensity Score Matching
# Dataset:  LaLonde (1986) — same as Phase 2, different identification strategy
# Estimand: ATT — Average Treatment Effect on the Treated
# Benchmark: ~$1,794 (LaLonde RCT)
#
# What Phase 2 (DiD) couldn't do that PSM does:
#   DiD needed panel data (pre + post observations per worker) and a
#   parallel trends assumption. PSM works on a single cross-section —
#   just 1978 earnings + covariates. It constructs a comparable control
#   group directly from covariate information rather than time variation.

# COMMAND ----------

# MAGIC %pip install scipy statsmodels scikit-learn mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
import mlflow
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

TRUE_ATT = 1794  # LaLonde RCT benchmark
print("✓ Imports complete")
print(f"  Benchmark ATT: ${TRUE_ATT:,}")

# COMMAND ----------

import numpy as np
import pandas as pd

TRUE_ATT = 1800

def make_group(n, treat, age_mean, educ_mean, pct_black, pct_hisp, pct_married, pct_nodegree,
               re74_mean, re75_mean, re74_sd, re75_sd, true_effect=0, earnings_noise=4000, seed=None):
    if seed is not None: np.random.seed(seed)
    re74 = np.random.normal(re74_mean, re74_sd, n).clip(0)
    re75 = np.random.normal(re75_mean, re75_sd, n).clip(0)
    re78 = (0.90 * re75 + 0.15 * re74 + true_effect + np.random.normal(0, earnings_noise, n)).clip(0)
    return pd.DataFrame({
        "treat": treat,
        "age": np.random.normal(age_mean, 7, n).clip(17, 60).round(),
        "educ": np.random.normal(educ_mean, 2, n).clip(0, 18).round(),
        "black": np.random.binomial(1, pct_black, n),
        "hisp": np.random.binomial(1, pct_hisp, n),
        "married": np.random.binomial(1, pct_married, n),
        "nodegree": np.random.binomial(1, pct_nodegree, n),
        "re74": re74.round(2),
        "re75": re75.round(2),
        "re78": re78.round(2),
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
        n=15992, treat=0, age_mean=28, educ_mean=11, pct_black=0.11, pct_hisp=0.08,
        pct_married=0.45, pct_nodegree=0.45, re74_mean=8725, re75_mean=7400,
        re74_sd=9000, re75_sd=8110, true_effect=0, seed=seed + 1
    )
    return df_treat, df_ctrl, df_cps

# COMMAND ----------

# =============================================================================
# Load LaLonde Data (same as Phase 2)
# =============================================================================
#
# KEY DIFFERENCE FROM PHASE 2:
# Phase 2 used re75 (pre-period) AND re78 (post-period) — panel structure.
# Phase 3 uses ONLY re78 (post-period outcome) + covariates.
# We deliberately ignore the panel structure to simulate a cross-sectional
# setting — showing PSM works without a pre-period.
#
# We still use re74/re75 as COVARIATES (pre-treatment characteristics that
# predict treatment selection), not as a panel outcome.

TREATED_URL = ("http://www.nber.org/~rdehejia/data/nswre74_treated.txt"
    # "https://raw.githubusercontent.com/robjellis/lalonde/master/"
    # "lalonde_treated.csv"
)

COLUMNS = ['treat', 'age', 'educ', 'black', 'hisp', 'married', 'nodegree', 're74', 're75', 're78']

CONTROL_URL = (
    "http://www.nber.org/~rdehejia/data/psid_controls.txt"
    # "https://raw.githubusercontent.com/robjellis/lalonde/master/"
    # "lalonde_control.csv"
)

CPS_CONTROL_URL = (
    # "http://www.nber.org/~rdehejia/data/cps2_controls.txt"
    "http://www.nber.org/~rdehejia/data/cps3_controls.txt"
)

try:
    df_treat = pd.read_csv(TREATED_URL, names=COLUMNS, sep = '  ')
    df_ctrl  = pd.read_csv(CONTROL_URL, names=COLUMNS, sep = '  ')
    df_cps_ctrl = pd.read_csv(CPS_CONTROL_URL, names=COLUMNS, sep = '  ')
    print("✓ Loaded from NBER website")
except Exception:
    # Fallback: simulate LaLonde-like data with published summary stats
    df_treat, df_ctrl, df_cps_ctrl = get_simulated_lalonde()
# Combine into one DataFrame

# Why invalid dataframe? Keep going, you'll know soon

df_invalid = pd.concat([df_treat, df_ctrl], ignore_index=True)
df_invalid['treat'] = df_invalid['treat'].astype(float).astype(int)

df = pd.concat([df_treat, df_cps_ctrl], ignore_index=True)
df['treat'] = df['treat'].astype(float).astype(int)

print(f"\nDataset shape: {df_invalid.shape}")
print(f"Treated workers:   {df_invalid['treat'].sum():,}")
print(f"Control workers:   {(df_invalid['treat']==0).sum():,}")
print(f"\nColumns: {list(df_invalid.columns)}")
# Covariates used for propensity score estimation
COVARIATES = ['age', 'educ', 'black', 'hisp', 'married', 'nodegree', 're74', 're75']
OUTCOME    = 're78'
TREATMENT  = 'treat'

print(f"\nDataset: {df_invalid.shape[0]:,} workers")
print(f"  Treated:  {df_invalid['treat'].sum():,}")
print(f"  Control:  {(df_invalid['treat']==0).sum():,}")
print(f"\nCovariates: {COVARIATES}")
print(f"Outcome:    {OUTCOME}")

# COMMAND ----------

# =============================================================================
# Naive Estimate + Baseline Imbalance
# =============================================================================
#
# Before any matching, establish:
#   (a) The naive (biased) estimate
#   (b) How imbalanced the groups are on covariates (the problem PSM solves)
#
# Standardized Mean Difference (SMD):
#   SMD = (mean_treated - mean_control) / pooled_std
#   |SMD| < 0.1 = good balance
#   |SMD| > 0.2 = meaningful imbalance

def compute_smd(df, covariates, treat_col='treat'):
    """Compute standardized mean difference for each covariate."""
    smds = {}
    for col in covariates:
        m1 = df[df[treat_col]==1][col].mean()
        m0 = df[df[treat_col]==0][col].mean()
        s  = df[col].std()
        smds[col] = ((m1 - m0) / s) if s > 0 else 0
    return pd.Series(smds)

def compare_naive(df, OUTCOME, COVARIATES, TRUE_ATT, treatment_col='treat'):
    """Compute naive estimate and compare to true ATT."""
    naive_att = (df[df[treatment_col]==1][OUTCOME].mean()
            - df[df[treatment_col]==0][OUTCOME].mean())
    print(f"\nNaive ATT (raw mean difference): ${naive_att:,.0f}")
    print(f"True ATT (benchmark):           ${TRUE_ATT:,}")

    print(f"\nCovariate SMD before matching:")
    print(f"{'Covariate':<12} {'SMD':>8}  {'Balance':>10}")
    print("─" * 34)

    smds_before = compute_smd(df, COVARIATES)

    for cov, smd in smds_before.items():
        status = "✓ OK" if abs(smd) < 0.1 else ("⚠ Moderate" if abs(smd) < 0.2 else "✗ Imbalanced")
        print(f"{cov:<12} {smd:>8.3f}  {status:>10}")

    print(f"\nMax |SMD|: {smds_before.abs().max():.3f}")
    print(f"Mean |SMD|: {smds_before.abs().mean():.3f}")
    print(f"\nConclusion: groups are severely imbalanced — direct comparison is invalid.")
    return naive_att, smds_before

# COMMAND ----------

# =============================================================================
# Estimate Propensity Scores
# =============================================================================
#
# The propensity score is: p(X) = P(treat=1 | X)
# We estimate this with logistic regression.
#
# Recall from our conceptual discussion:
#   - We care about CALIBRATION, not accuracy
#   - A MODEL THAT PERFECTLY SEPARATES TREATED FROM CONTROL IS BAD
#     (means no overlap — matching becomes impossible) read blog for explanation
#   - Logistic regression is naturally calibrated
#
# We include ALL available covariates — including re74 and re75.
# These are pre-treatment earnings and are legitimate confounders:
# they affect both who enrolls in training AND future earnings.

def estimate_propensity(df, covariates, treatment_col='treat'):
    """Estimate propensity scores using logistic regression."""
    X = df[COVARIATES].values
    y = df[treatment_col].values

    # Standardize features — logistic regression is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)

    # Propensity scores — probability of being treated
    df['pscore'] = lr.predict_proba(X_scaled)[:, 1]

    # Model diagnostics
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, df['pscore'])

    print(f"\nLogistic regression fitted.")
    print(f"  AUC: {auc:.3f}")
    print(f"\n  NOTE: AUC should be moderate (0.6–0.85).")
    print(f"  Too high (>0.95) → near-perfect separation → no overlap → bad")
    print(f"  Too low (<0.55)  → covariates don't predict treatment → check data")

    print(f"\nPropensity score summary:")
    print(df.groupby('treat')['pscore'].describe().round(3).to_string())

    # Check overlap: do treated and control p-scores share a common range?
    treated_ps  = df[df[treatment_col]==1]['pscore']
    control_ps  = df[df[treatment_col]==0]['pscore']
    overlap_min = max(treated_ps.min(), control_ps.min())
    overlap_max = min(treated_ps.max(), control_ps.max())

    print(f"\nCommon support (overlap) region: [{overlap_min:.3f}, {overlap_max:.3f}]")
    n_treated_in_support = ((treated_ps >= overlap_min) & (treated_ps <= overlap_max)).sum()
    print(f"Treated units in support: {n_treated_in_support} / {len(treated_ps)}")

    # AUC-based overlap warning
    if auc > 0.85:
        print(f"\n⚠ WARNING: AUC = {auc:.3f} — near-perfect separation detected")
        print(f"  This means treated and control groups are fundamentally different")
        print(f"  on observed covariates. PSM will have very limited common support.")
        print(f"  Options: (1) use a better comparison group, (2) trim to common")
        print(f"  support, (3) consider IPW or DML instead of PSM.")
        print("Luckily we have our CPS control to continue learning PSM",
              "(option 1: better comparision group)")
    return df, treated_ps, control_ps, overlap_min, overlap_max

# COMMAND ----------

# =============================================================================
# Visualize Propensity Score Overlap
# =============================================================================
#
# The overlap plot is the most important diagnostic for PSM.
# It shows whether treated and control units share common support —
# i.e., whether matching is actually possible.
#
# If the distributions don't overlap at all, PSM is invalid.
# You'd be extrapolating — matching treated units to controls
# that look nothing like them.

def visualize_propensity_estimate(treated_ps, control_ps, overlap_min, overlap_max, save_fig = False):
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    fig.suptitle("Phase 3 — Propensity Score Overlap", fontsize=13, fontweight='bold')

    # ── Plot A: Histogram overlap ────────────────────────────────────────────────
    ax.hist(control_ps,  bins=50, alpha=0.55, color='#5b8dd9',
            label=f'Control (n={len(control_ps):,})', density=True)
    ax.hist(treated_ps,  bins=50, alpha=0.70, color='#e06b5b',
            label=f'Treated (n={len(treated_ps):,})', density=True)
    ax.axvline(overlap_min, color='#333', linestyle='--', linewidth=1.2,
            label=f'Common support [{overlap_min:.2f}, {overlap_max:.2f}]')
    ax.axvline(overlap_max, color='#333', linestyle='--', linewidth=1.2)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity score distributions\n(overlap = matching is possible)",
                fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # Annotate overlap region
    ax.axvspan(overlap_min, overlap_max, alpha=0.08, color='green',
            label='Overlap region')

    plt.tight_layout()
    if save_fig:
        plt.savefig('/tmp/phase3_overlap.png', dpi=150, bbox_inches='tight')
        print("✓ Overlap plot saved")
    plt.show()

# COMMAND ----------

# =============================================================================
# Invalid Comparison Group Diagnostic
# =============================================================================
#
# compare_naive: Shows the naive ATT estimate (raw mean difference)
#                and quantifies baseline covariate imbalance using
#                Standardized Mean Differences (SMD). Highlights
#                that groups are severely imbalanced and direct
#                comparison is invalid.
#
# estimate_propensity: Fits logistic regression to estimate propensity scores.
#                      Prints AUC and overlap diagnostics. Reveals near-perfect
#                      separation (high AUC), meaning treated and control groups
#                      are fundamentally different on observed covariates.
#                      Warns that PSM will have very limited common support.

_ = compare_naive(df_invalid, OUTCOME, COVARIATES, TRUE_ATT)

df_invalid, treated_ps, control_ps, overlap_min, overlap_max = estimate_propensity(df_invalid, COVARIATES)

visualize_propensity_estimate(treated_ps, control_ps, overlap_min, overlap_max)

# NOTE: Minimal overlap between treated and control groups.
#       Will not be able to find matching (similar) units for PSM.
#       This demonstrates why direct comparison is invalid and why
#       a better control group is needed.

# COMMAND ----------

# =============================================================================
# (~) Valid Comparison Group Diagnostic
# =============================================================================
#
# compare_naive: Shows the naive ATT estimate (raw mean difference)
#                and quantifies baseline covariate imbalance using
#                Standardized Mean Differences (SMD). Highlights
#                that groups are severely imbalanced and direct
#                comparison is invalid.
#
# estimate_propensity: Fits logistic regression to estimate propensity scores.
#                      Prints AUC and overlap diagnostics. Reveals near-perfect
#                      separation (high AUC), meaning treated and control groups
#                      are fundamentally different on observed covariates.
#                      Warns that PSM will have very limited common support.

naive_att, smds_before = compare_naive(df, OUTCOME, COVARIATES, TRUE_ATT)

df, treated_ps, control_ps, overlap_min, overlap_max = estimate_propensity(df, COVARIATES)

visualize_propensity_estimate(treated_ps, control_ps, overlap_min, overlap_max)

# COMMAND ----------

# =============================================================================
# Nearest Neighbor Matching with Caliper
# =============================================================================
#
# Algorithm:
#   For each treated unit:
#     1. Find the control unit with the closest propensity score
#     2. Only accept the match if |ps_treated - ps_control| < caliper
#     3. Match WITHOUT replacement — each control used at most once
#
# CALIPER: Maximum allowed propensity score distance for a match.
# Standard choice: 0.2 × std(propensity score) — Rosenbaum & Rubin (1985)
#
# Why caliper? Without it, treated units in sparse regions get matched
# to very different controls — bad matches inflate bias.
# 
# Why without replacement? With replacement, the same control unit
# matches many treated units — the effective sample size shrinks and
# standard errors are underestimated.

# Standard caliper: 0.2 × SD of propensity score
ps_sd     = df['pscore'].std()
caliper   = 0.2 * ps_sd
print(f"\nPS standard deviation: {ps_sd:.4f}")
print(f"Caliper (0.2 × SD):    {caliper:.4f}")

# Separate treated and control
treated_idx = df[df['treat']==1].index.tolist()
control_idx = df[df['treat']==0].index.tolist()

treated_ps_arr = df.loc[treated_idx, 'pscore'].values.reshape(-1, 1)
control_ps_arr = df.loc[control_idx, 'pscore'].values.reshape(-1, 1)

# Fit nearest neighbor on control propensity scores
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control_ps_arr)

# Find nearest control for each treated unit
distances, indices = nn.kneighbors(treated_ps_arr)
distances = distances.flatten()
indices   = indices.flatten()

# Apply caliper — discard matches where distance > caliper
matched_pairs = []
used_controls = set()

for i, (dist, ctrl_pos) in enumerate(zip(distances, indices)):
    ctrl_idx = control_idx[ctrl_pos]
    if dist <= caliper and ctrl_idx not in used_controls:
        matched_pairs.append({
            'treated_idx': treated_idx[i],
            'control_idx': ctrl_idx,
            'ps_distance': dist
        })
        used_controls.add(ctrl_idx)

pairs_df = pd.DataFrame(matched_pairs)
n_matched = len(pairs_df)
n_unmatched = len(treated_idx) - n_matched

print(f"\nMatching results:")
print(f"  Treated units:           {len(treated_idx):,}")
print(f"  Matched pairs:           {n_matched:,}")
print(f"  Unmatched (caliper):     {n_unmatched:,}  ← dropped, too different")
print(f"  Mean PS distance:        {pairs_df['ps_distance'].mean():.5f}")
print(f"  Max PS distance:         {pairs_df['ps_distance'].max():.5f}")

# Build matched sample DataFrame
treated_matched = df.loc[pairs_df['treated_idx']].copy()
control_matched = df.loc[pairs_df['control_idx']].copy()
df_matched = pd.concat([treated_matched, control_matched], ignore_index=True)

print(f"\nMatched sample size: {len(df_matched):,} ({n_matched} treated + {n_matched} control)")

# COMMAND ----------

# =============================================================================
# Covariate Balance After Matching (Love Plot)
# =============================================================================
#
# After matching, check whether covariates are now balanced.
# The Love plot shows SMD for each covariate BEFORE and AFTER matching.
#
# Target: |SMD| < 0.1 for all covariates after matching.
# If some covariates are still imbalanced, consider:
#   - Adding more covariates to the PS model
#   - Using a tighter caliper
#   - Using full matching or kernel matching instead

smds_after = compute_smd(df_matched, COVARIATES)

balance_df = pd.DataFrame({
    'Before': smds_before.abs(),
    'After':  smds_after.abs()
})

print(f"\n{'Covariate':<12} {'SMD Before':>12} {'SMD After':>10} {'Improved':>10}")
print("─" * 46)
for cov in COVARIATES:
    before = smds_before[cov]
    after  = smds_after[cov]
    improved = "✓" if abs(after) < abs(before) else "✗"
    ok = "✓ OK" if abs(after) < 0.1 else "⚠"
    print(f"{cov:<12} {before:>12.3f} {after:>10.3f} {improved:>5} {ok}")

print(f"\nMean |SMD| before: {smds_before.abs().mean():.3f}")
print(f"Mean |SMD| after:  {smds_after.abs().mean():.3f}")
print(f"Reduction:         {(1 - smds_after.abs().mean()/smds_before.abs().mean())*100:.1f}%")

# ── Love Plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

y_pos = np.arange(len(COVARIATES))
ax.scatter(balance_df['Before'], y_pos, color='#e06b5b', s=80,
           zorder=3, label='Before matching', marker='o')
ax.scatter(balance_df['After'],  y_pos, color='#5b8dd9', s=80,
           zorder=3, label='After matching',  marker='D')

# Connect before and after for each covariate
for i in y_pos:
    ax.plot([balance_df['Before'].iloc[i], balance_df['After'].iloc[i]],
            [i, i], color='#ccc', linewidth=1.5, zorder=2)

ax.axvline(0.1, color='#333', linestyle='--', linewidth=1.2,
           label='SMD = 0.1 threshold')
ax.axvline(0.0, color='#888', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(COVARIATES)
ax.set_xlabel("Absolute Standardized Mean Difference (SMD)")
ax.set_title("Love Plot — Covariate Balance Before and After Matching",
             fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(-0.02, max(balance_df['Before'].max(), 0.25) * 1.1)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/phase3_love_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Love plot saved")

# COMMAND ----------

# =============================================================================
# ATT Estimation on Matched Sample
# =============================================================================
#
# Now that we have matched pairs, the ATT estimate is straightforward:
#   ATT = mean(Y_treated) - mean(Y_control) over matched pairs
#
# This works because the matched pairs are (approximately) balanced
# on all observed covariates — the control units are valid counterfactuals
# for the treated units.
#
# We also run OLS on the matched sample with covariates for additional
# precision. Because balance is good, the OLS estimate should be close
# to the simple mean difference.

# --- Simple mean difference on matched sample ---
y1_matched = df_matched[df_matched['treat']==1][OUTCOME]
y0_matched = df_matched[df_matched['treat']==0][OUTCOME]
ATT_simple = y1_matched.mean() - y0_matched.mean()

# Standard error via paired differences
diffs  = (treated_matched[OUTCOME].values
        - control_matched[OUTCOME].values)
SE_paired = diffs.std() / np.sqrt(len(diffs))
CI_low  = ATT_simple - 1.96 * SE_paired
CI_high = ATT_simple + 1.96 * SE_paired
t_stat  = ATT_simple / SE_paired
p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs)-1))

print(f"\nSimple mean difference (matched pairs):")
print(f"  ATT estimate:  ${ATT_simple:,.2f}")
print(f"  Std Error:     ${SE_paired:,.2f}")
print(f"  95% CI:        [${CI_low:,.0f},  ${CI_high:,.0f}]")
print(f"  t-statistic:   {t_stat:.3f}")
print(f"  p-value:       {p_val:.4f}")

# --- OLS on matched sample with covariates ---
formula = f"{OUTCOME} ~ treat + age + educ + black + hisp + married + nodegree + re74 + re75"
ols_matched = smf.ols(formula, data=df_matched).fit(cov_type='HC3')
ATT_ols = ols_matched.params['treat']
CI_ols  = ols_matched.conf_int().loc['treat']
SE_ols  = ols_matched.bse['treat']

print(f"\nOLS on matched sample (+ covariates, HC3 SE):")
print(f"  ATT estimate:  ${ATT_ols:,.2f}")
print(f"  Std Error:     ${SE_ols:,.2f}")
print(f"  95% CI:        [${CI_ols[0]:,.0f},  ${CI_ols[1]:,.0f}]")
print(f"  p-value:       {ols_matched.pvalues['treat']:.4f}")

# --- Benchmark comparison ---
print(f"\n{'─'*60}")
print(f"  True ATT (RCT):           ${TRUE_ATT:,}")
print(f"  Naive estimate:           ${naive_att:,.0f}   ← biased")
print(f"  PSM simple:               ${ATT_simple:,.0f}")
print(f"  PSM + OLS covariates:     ${ATT_ols:,.0f}")
print(f"\n  Recovery (PSM simple):    {ATT_simple/TRUE_ATT*100:.1f}% of true ATT")
print(f"  Recovery (PSM + OLS):     {ATT_ols/TRUE_ATT*100:.1f}% of true ATT")

# COMMAND ----------

# =============================================================================
# CELL 9 — Sensitivity Analysis (Rosenbaum Bounds)
# =============================================================================
#
# PSM's critical assumption: no UNMEASURED confounders.
# We can never verify this — we can only ask: how sensitive is our
# estimate to potential unmeasured confounding?
#
# Rosenbaum's sensitivity analysis asks:
#   "How strong would an unmeasured confounder have to be to explain
#    away the entire estimated treatment effect?"
#
# The parameter Γ (gamma) represents how much more likely a treated
# unit is to be treated than a matched control, due to an unmeasured
# confounder. Γ=1 means no hidden bias (our base assumption).
#
# We compute bounds on the p-value for different values of Γ.
# If our effect remains significant even at Γ=2 (unmeasured confounder
# doubles the odds of treatment), the result is robust.

print("=" * 60)
print("SENSITIVITY ANALYSIS — Rosenbaum Bounds")
print("=" * 60)
print("""
Γ = 1: No unmeasured confounding (our assumption)
Γ = 1.5: Hidden variable makes treated 1.5× more likely to be treated
Γ = 2: Hidden variable makes treated 2× more likely to be treated
Γ = 3: Hidden variable makes treated 3× more likely to be treated

For each Γ, we compute the WORST-CASE p-value (upper bound).
If worst-case p < 0.05, result is significant even under that level
of hidden confounding.
""")

def rosenbaum_bound(diffs, gamma):
    """
    Simplified Rosenbaum bounds for matched pair differences.
    Returns the worst-case (upper bound) p-value under gamma.
    """
    n = len(diffs)
    # Under no treatment effect null, with gamma-level hidden bias,
    # each pair's probability of the treated being higher is at most
    # gamma/(1+gamma)
    p_upper = gamma / (1 + gamma)

    # Wilcoxon signed rank statistic on positive differences
    pos_diffs = diffs[diffs > 0]
    T_plus = len(pos_diffs)

    # Under null with p_upper, T+ ~ Binomial(n, p_upper)
    # Compute upper bound p-value
    from scipy.stats import binom
    p_val_upper = 1 - binom.cdf(T_plus - 1, n, p_upper)
    return p_val_upper

gammas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
print(f"{'Γ':>6}  {'Upper-bound p-value':>22}  {'Significant at 5%':>18}")
print("─" * 52)
for g in gammas:
    pval = rosenbaum_bound(diffs, g)
    sig  = "✓ Yes" if pval < 0.05 else "✗ No"
    print(f"{g:>6.2f}  {pval:>22.5f}  {sig:>18}")

print(f"\nConclusion: Our result remains significant (p<0.05) up to")
critical_g = None
for g in gammas:
    if rosenbaum_bound(diffs, g) < 0.05:
        critical_g = g
if critical_g:
    print(f"  Γ ≈ {critical_g:.2f} — an unmeasured confounder would need to more than")
    print(f"  double the odds of treatment to explain away the effect.")
print(f"\nThis is a quantitative answer to 'but what about unmeasured confounders?'")

# COMMAND ----------

# =============================================================================
# CELL 10 — Full Results Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("Phase 3 — Propensity Score Matching Results",
             fontsize=13, fontweight='bold')

# ── Plot A: Distribution of matched pair differences ─────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.hist(diffs, bins=40, color='#7fb3d3', edgecolor='white', linewidth=0.3)
ax.axvline(ATT_simple, color='#e06b5b', linewidth=2,
           label=f'ATT = ${ATT_simple:,.0f}')
ax.axvline(0, color='#333', linewidth=0.8, linestyle='--')
ax.axvline(CI_low,  color='#888', linewidth=1.2, linestyle=':')
ax.axvline(CI_high, color='#888', linewidth=1.2, linestyle=':',
           label=f'95% CI: [${CI_low:,.0f}, ${CI_high:,.0f}]')
ax.set_xlabel("Outcome difference (treated − control)")
ax.set_ylabel("Count")
ax.set_title("Matched pair\noutcome differences", fontweight='bold')
ax.legend(fontsize=7)
ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))

# ── Plot B: Propensity score distributions after matching ────────────────────
ax = fig.add_subplot(gs[0, 1])
t_ps_m = df_matched[df_matched['treat']==1]['pscore']
c_ps_m = df_matched[df_matched['treat']==0]['pscore']
ax.hist(c_ps_m, bins=30, alpha=0.55, color='#5b8dd9',
        label='Control (matched)', density=True)
ax.hist(t_ps_m, bins=30, alpha=0.7,  color='#e06b5b',
        label='Treated (matched)', density=True)
ax.set_xlabel("Propensity Score")
ax.set_ylabel("Density")
ax.set_title("PS distributions\nafter matching", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

# ── Plot C: Estimate comparison ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
methods   = ['True ATT\n(RCT)', 'Naive\ndiff', 'PSM\nsimple', 'PSM\n+OLS']
estimates = [TRUE_ATT, naive_att, ATT_simple, ATT_ols]
colors    = ['#2ecc71', '#e74c3c', '#3498db', '#2980b9']
bars = ax.bar(methods, estimates, color=colors, width=0.5, alpha=0.85)
for bar, val in zip(bars, estimates):
    yoff = 80 if val >= 0 else -250
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + yoff,
            f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(TRUE_ATT, color='#2ecc71', linestyle='--',
           linewidth=1.5, alpha=0.7, label=f'True ATT = ${TRUE_ATT:,}')
ax.axhline(0, color='#333', linewidth=0.8)
ax.set_title("Estimates vs benchmark", fontweight='bold')
ax.set_ylabel("ATT estimate ($)")
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

# ── Plot D: Rosenbaum bounds ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
gamma_vals = np.linspace(1.0, 3.5, 50)
pvals_upper = [rosenbaum_bound(diffs, g) for g in gamma_vals]
ax.plot(gamma_vals, pvals_upper, color='#e06b5b', linewidth=2)
ax.axhline(0.05, color='#333', linestyle='--', linewidth=1.2,
           label='p = 0.05')
ax.fill_between(gamma_vals, pvals_upper, 0.05,
                where=[p < 0.05 for p in pvals_upper],
                alpha=0.15, color='#2ecc71', label='Still significant')
ax.set_xlabel("Γ (hidden bias parameter)")
ax.set_ylabel("Worst-case p-value")
ax.set_title("Rosenbaum sensitivity\nbounds", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

# ── Plot E: SMD comparison before/after ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 1:])
y_pos = np.arange(len(COVARIATES))
ax.barh(y_pos - 0.2, smds_before.abs(), height=0.35,
        color='#e06b5b', alpha=0.75, label='Before matching')
ax.barh(y_pos + 0.2, smds_after.abs(),  height=0.35,
        color='#5b8dd9', alpha=0.75, label='After matching')
ax.axvline(0.1, color='#333', linestyle='--', linewidth=1.2,
           label='SMD = 0.1 threshold')
ax.set_yticks(y_pos)
ax.set_yticklabels(COVARIATES)
ax.set_xlabel("Absolute SMD")
ax.set_title("Covariate balance before vs after matching",
             fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.savefig('/tmp/phase3_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Results plot saved")

# COMMAND ----------

# =============================================================================
# CELL 11 — MLflow Logging
# =============================================================================

print("=" * 60)
print("MLFLOW LOGGING")
print("=" * 60)

mlflow.set_experiment("causal_inference_toolkit")

with mlflow.start_run(run_name="phase3_psm"):

    # Parameters
    mlflow.log_param("method",           "PSM")
    mlflow.log_param("dataset",          "LaLonde_PSID")
    mlflow.log_param("estimand",         "ATT")
    mlflow.log_param("ps_model",         "LogisticRegression")
    mlflow.log_param("matching",         "NearestNeighbor_1to1_no_replacement")
    mlflow.log_param("caliper",          round(caliper, 5))
    mlflow.log_param("n_treated",        len(treated_idx))
    mlflow.log_param("n_control_raw",    len(control_idx))
    mlflow.log_param("n_matched_pairs",  n_matched)
    mlflow.log_param("n_unmatched",      n_unmatched)
    mlflow.log_param("true_att",         TRUE_ATT)
    mlflow.log_param("covariates",       str(COVARIATES))

    # Metrics
    mlflow.log_metric("ATT_simple",       round(ATT_simple, 2))
    mlflow.log_metric("ATT_ols",          round(ATT_ols, 2))
    mlflow.log_metric("SE_paired",        round(SE_paired, 2))
    mlflow.log_metric("SE_ols",           round(SE_ols, 2))
    mlflow.log_metric("CI_lower",         round(CI_low, 2))
    mlflow.log_metric("CI_upper",         round(CI_high, 2))
    mlflow.log_metric("p_value",          round(p_val, 5))
    mlflow.log_metric("ps_auc",           round(auc, 4))
    mlflow.log_metric("caliper",          round(caliper, 5))
    mlflow.log_metric("smd_mean_before",  round(smds_before.abs().mean(), 4))
    mlflow.log_metric("smd_mean_after",   round(smds_after.abs().mean(), 4))
    mlflow.log_metric("smd_reduction_pct",
                      round((1 - smds_after.abs().mean()/smds_before.abs().mean())*100, 1))
    mlflow.log_metric("naive_att",        round(naive_att, 2))
    mlflow.log_metric("recovery_pct",
                      round(ATT_simple/TRUE_ATT*100, 1))

    # Artifacts
    mlflow.log_artifact('/tmp/phase3_overlap.png')
    mlflow.log_artifact('/tmp/phase3_love_plot.png')
    mlflow.log_artifact('/tmp/phase3_results.png')

    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ MLflow run logged — Run ID: {run_id}")
    print(f"  ATT (simple):      ${ATT_simple:,.2f}")
    print(f"  ATT (OLS):         ${ATT_ols:,.2f}")
    print(f"  95% CI:            [${CI_low:,.0f}, ${CI_high:,.0f}]")
    print(f"  SMD reduction:     {(1 - smds_after.abs().mean()/smds_before.abs().mean())*100:.1f}%")
    print(f"  Recovery:          {ATT_simple/TRUE_ATT*100:.1f}% of true ATT")

# COMMAND ----------

# =============================================================================
# CELL 12 — Summary and Bridge to Phase 4 (IPW)
# =============================================================================

print("=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)
print(f"""
What we did:
  ✓ Established baseline imbalance (mean |SMD| = {smds_before.abs().mean():.3f})
  ✓ Estimated propensity scores via logistic regression (AUC = {auc:.3f})
  ✓ Visualized overlap — common support region confirmed
  ✓ 1:1 nearest-neighbor matching with caliper = {caliper:.4f}
  ✓ Matched {n_matched} pairs ({n_unmatched} treated unmatched)
  ✓ Built Love plot — covariate balance after matching
  ✓ Estimated ATT via paired differences and OLS
  ✓ Rosenbaum sensitivity bounds — quantified hidden bias robustness
  ✓ Logged all results to MLflow

Key results:
  True ATT (RCT):          ${TRUE_ATT:,}
  Naive estimate:          ${naive_att:,.0f}   ← biased
  PSM simple:              ${ATT_simple:,.0f}
  PSM + OLS:               ${ATT_ols:,.0f}
  Mean |SMD| before:       {smds_before.abs().mean():.3f}
  Mean |SMD| after:        {smds_after.abs().mean():.3f}  ← much better

What PSM left unsolved → motivates Phase 4 (IPW):
  PSM DISCARDED {n_unmatched} treated units — those without a close enough
  control match. This means:
    (a) We're estimating ATT on a SUBSET of treated workers, not all of them
    (b) Data loss reduces statistical power
    (c) The estimand changed slightly — we're now estimating the ATT
        only for matchable treated units

  IPW (Inverse Probability Weighting) fixes this. Instead of discarding
  units, it REWEIGHTS every observation. Treated units with low propensity
  scores (surprising — they got treated despite low probability) are
  upweighted. Control units with high propensity scores (they looked like
  they should have been treated but weren't) are also upweighted.

  Result: every observation contributes to the estimate. No data loss.
  And we can estimate ATE (not just ATT) — the effect on the full
  population, not just on those who were treated.
""")
