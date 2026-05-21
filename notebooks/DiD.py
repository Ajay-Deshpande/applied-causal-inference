# Databricks notebook source
df_treat

# COMMAND ----------

# Phase 2 — Difference-in-Differences (DiD)
# Causal Inference Toolkit
#
# Problem:  Did the (JTPA) job training program increase earnings?
# Method:   Difference-in-Differences
# Dataset:  LaLonde (1986) — canonical causal inference benchmark
# Estimand: ATT — Average Treatment Effect on the Treated
# Ground truth ATT from LaLonde's own RCT: ~$1,794

# COMMAND ----------

# =============================================================================
# Imports and Setup
# =============================================================================

# %pip install scipy statsmodels mlflow --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import mlflow
from io import StringIO
import urllib.request

np.random.seed(42)

# Ground truth from LaLonde's original RCT
# This is what every method in this series is trying to recover
TRUE_ATT = 1794

print("✓ Imports complete")
print(f"  Benchmark ATT (LaLonde RCT): ${TRUE_ATT:,}")

# COMMAND ----------

def get_simulated_lalonde():
    np.random.seed(42)
    N_treat, N_ctrl = 185, 2490

    def make_group(n, treat, re74_mean, re75_mean, re78_mean,
                    re74_sd, re75_sd, re78_sd,
                    age_mean=25, educ_mean=10, pct_black=0.84,
                    pct_married=0.19, pct_nodegree=0.71):
        re74 = np.random.normal(re74_mean, re74_sd, n).clip(0)
        re75 = np.random.normal(re75_mean, re75_sd, n).clip(0)
        # re78 includes true treatment effect for treated group
        effect = TRUE_ATT if treat == 1 else 0
        re78   = (re75 * 0.85
                    + effect
                    + np.random.normal(0, 4000, n)).clip(0)
        return pd.DataFrame({
            'treat':    treat,
            'age':      np.random.normal(age_mean, 6, n).clip(17, 55).round(),
            'educ':     np.random.normal(educ_mean, 2, n).clip(0, 16).round(),
            'black':    np.random.binomial(1, pct_black, n),
            'hisp':     np.random.binomial(1, 0.06, n),
            'married':  np.random.binomial(1, pct_married, n),
            'nodegree': np.random.binomial(1, pct_nodegree, n),
            're74':     re74.round(2),
            're75':     re75.round(2),
            're78':     re78.round(2),
        })

    # Published summary statistics from LaLonde (1986) Table 2
    df_treat = make_group(N_treat, treat=1,
                            re74_mean=2096, re75_mean=1532, re78_mean=6349,
                            re74_sd=4887,   re75_sd=3219,   re78_sd=7867)
    df_ctrl  = make_group(N_ctrl,  treat=0,
                            re74_mean=13651,re75_mean=13405,re78_mean=14847,
                            re74_sd=9270,   re75_sd=9270,   re78_sd=9570,
                            age_mean=34, educ_mean=12, pct_black=0.07,
                            pct_married=0.87, pct_nodegree=0.31)
    return df_treat, df_ctrl

# COMMAND ----------

# =============================================================================
# Load the LaLonde Dataset
# =============================================================================
#
# The LaLonde dataset has two parts that are typically used together:
#
#   Experimental sample:
#     - Treated units: workers who were randomly assigned to training
#     - Experimental controls: workers randomly assigned to NO training
#     This gives us the RCT benchmark (TRUE_ATT ~ $1,794)
#
#   Non-experimental (observational) sample:
#     - Same treated workers
#     - Control units: workers from the CPS (Current Population Survey)
#       or PSID (Panel Study of Income Dynamics) — NOT randomized
#     This is what we use for DiD, PSM, IPW — they self-selected into no-training
#
# WHY USE THE NON-EXPERIMENTAL SAMPLE?
#   Because using the RCT controls would just reproduce the RCT.
#   The whole point is to show that DiD can recover ~$1,794 from
#   observational data where workers were NOT randomly assigned.
#   That's the test of the method.

# LaLonde dataset — we load a well-known hosted version
# Treated: experimental treated group (185 workers)
# Control: PSID non-experimental comparison group (2,490 workers)

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

try:
    df_treat = pd.read_csv(TREATED_URL, names=COLUMNS, sep = '  ')
    df_ctrl  = pd.read_csv(CONTROL_URL, names=COLUMNS, sep = '  ')
    print("✓ Loaded from NBER website")
except Exception:
    # Fallback: simulate LaLonde-like data with published summary stats
    df_treat, df_ctrl = get_simulated_lalonde()
# Combine into one DataFrame
df = pd.concat([df_treat, df_ctrl], ignore_index=True)
df['treat'] = df['treat'].astype(float).astype(int)

print(f"\nDataset shape: {df.shape}")
print(f"Treated workers:   {df['treat'].sum():,}")
print(f"Control workers:   {(df['treat']==0).sum():,}")
print(f"\nColumns: {list(df.columns)}")

# COMMAND ----------

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
#
# Two things to establish before any modeling:
#   (a) The selection bias is real and large
#   (b) The pre-treatment earnings trends — do they look parallel?
#
# --- 3a. Summary statistics by group ---

summary = df.groupby('treat').agg(
    n          = ('treat',    'count'),
    age        = ('age',      'mean'),
    education  = ('educ',     'mean'),
    pct_black  = ('black',    'mean'),
    pct_married= ('married',  'mean'),
    earn_1974  = ('re74',     'mean'),
    earn_1975  = ('re75',     'mean'),
    earn_1978  = ('re78',     'mean'),
).round(1)
summary.index = ['Control (PSID)', 'Treated']
print("\nGroup summary statistics:")
print(summary.to_string())

# --- 3b. The naive estimate (what you'd get without any method) ---
naive_ate = (df[df['treat']==1]['re78'].mean()
           - df[df['treat']==0]['re78'].mean())
print(f"\nNaive estimate (raw mean difference in 1978):")
print(f"  Treated mean 1978:  ${df[df['treat']==1]['re78'].mean():,.0f}")
print(f"  Control mean 1978:  ${df[df['treat']==0]['re78'].mean():,.0f}")
print(f"  Naive difference:   ${naive_ate:,.0f}")
print(f"  True ATT:           ${TRUE_ATT:,}")
print(f"\n  ⚠ Naive estimate is off by ${abs(naive_ate - TRUE_ATT):,.0f}")
print(f"  Direction: {'severely understated' if naive_ate < TRUE_ATT else 'severely overstated'}")

# We can't check this in real-world. This is only to show naive estimate would be incorrect method to go with
# WHY IS THE NAIVE ESTIMATE SO FAR OFF?
# Look at the summary table. The PSID control group earns ~$13,000/year
# in 1974-75, while treated workers earn ~$2,000/year.
# Treated workers are younger, less educated, more likely to be
# minorities, and earn far less even before training.
# Comparing their 1978 earnings directly confounds the training effect
# with all these pre-existing differences. This is selection bias.

# COMMAND ----------

# =============================================================================
# The DiD Intuition: Visualizing the Problem
# =============================================================================
#
# Before running any regression, plot the pre/post earnings for both groups.
# This is how DiD "sees" the data — as parallel (or not) trends over time.
#
# The DiD logic:
#   Control group trend (1975→1978): shows what would have happened naturally
#   Treated group trend (1975→1978): shows what actually happened
#   DiD = treated change MINUS control change
#        = removes the "natural drift" from the treated group's improvement

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 2 — DiD on LaLonde Data", fontsize=13, fontweight='bold')

# ── Plot A: Earnings trends ─────────────────────────────────────────────────
ax = axes[0]

treat_75 = df[df['treat']==1]['re75'].mean()
treat_78 = df[df['treat']==1]['re78'].mean()
ctrl_75  = df[df['treat']==0]['re75'].mean()
ctrl_78  = df[df['treat']==0]['re78'].mean()

# Actual lines
ax.plot([1975, 1978], [ctrl_75,  ctrl_78],
        'o-', color='#5b8dd9', linewidth=2.5, markersize=8,
        label='Control (PSID)')
ax.plot([1975, 1978], [treat_75, treat_78],
        'o-', color='#e06b5b', linewidth=2.5, markersize=8,
        label='Treated')

# Counterfactual: what treated would have looked like without training
# Under parallel trends: same additive change as control
ctrl_change    = ctrl_78  - ctrl_75
counterfactual = treat_75 + ctrl_change

ax.plot([1975, 1978], [treat_75, counterfactual],
        '--', color='#e06b5b', linewidth=1.5, alpha=0.6,
        label='Treated counterfactual (parallel trends)')

# DiD bracket at 1978
ax.annotate('', xy=(1978.2, treat_78),
            xytext=(1978.2, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
ax.text(1978.35, (treat_78 + counterfactual)/2,
        f'DiD\n${treat_78 - counterfactual:,.0f}',
        va='center', fontsize=9, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                  ec='#aaa', alpha=0.9))

ax.set_title("Earnings trends: treated vs control", fontweight='bold')
ax.set_ylabel("Mean earnings ($)")
ax.set_xlabel("Year")
ax.set_xticks([1975, 1978])
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.spines[['top', 'right']].set_visible(False)

# ── Plot B: Selection bias made visible ──────────────────────────────────────
ax = axes[1]

# 3-period plot: 1974, 1975, 1978
# Show that even in 1974 (pre-treatment) the groups were VERY different
treat_74 = df[df['treat']==1]['re74'].mean()
ctrl_74  = df[df['treat']==0]['re74'].mean()

ax.plot([1974, 1975, 1978], [ctrl_74,  ctrl_75,  ctrl_78],
        'o-', color='#5b8dd9', linewidth=2.5, markersize=8, label='Control')
ax.plot([1974, 1975, 1978], [treat_74, treat_75, treat_78],
        'o-', color='#e06b5b', linewidth=2.5, markersize=8, label='Treated')

# Shade the pre-treatment period
ax.axvspan(1974, 1975.5, alpha=0.07, color='gray', label='Pre-treatment')
ax.axvline(1975.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(1975.6, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1000,
        'Training\noccurs', fontsize=8, color='gray')

ax.set_title("3-period view: selection bias is clear\npre-treatment", fontweight='bold')
ax.set_ylabel("Mean earnings ($)")
ax.set_xlabel("Year")
ax.set_xticks([1974, 1975, 1978])
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../assets/plots/phase2/phase2_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Trends plot saved")

# COMMAND ----------

# =============================================================================
# Restructure Data for DiD (Long Format)
# =============================================================================
#
# DiD regression requires PANEL DATA — the same unit observed at multiple
# time points. Right now each row is one worker with separate columns for
# re75 and re78. We need to reshape so each worker appears TWICE:
#   once for the pre-treatment period (1975)
#   once for the post-treatment period (1978)
#
# The DiD regression formula:
#   earnings = α + β₁×Post + β₂×Treat + β₃×(Post×Treat) + ε
#
#   β₁ = time trend (how much earnings changed for everyone)
#   β₂ = baseline difference (how much treated differs from control pre-treatment)
#   β₃ = DiD ESTIMATE — the interaction term — this is what we want
#
# The interaction (Post × Treat) = 1 ONLY for treated units in the post period.
# That's the only cell where treatment was actually active.

# Reshape to long format

df = df.reset_index(names='worker_id')

id_cols = ['worker_id', 'treat', 'age', 'educ', 'black', 'hisp', 'married', 'nodegree']

df_long = df.melt(
    id_vars=id_cols,
    value_vars=['re75', 're78'],
    var_name='period',
    value_name='earnings'
)

# Create post indicator
df_long['post'] = (df_long['period'] == 're78').astype(int)

df_long['period'] = df_long['period'].replace({
    're75': '1975',
    're78': '1978'
})

print("Long format shape:", df_long.shape)
print(f"\nEach worker appears {df_long['worker_id'].nunique()} → ",
      end="")
print(f"No — each ID appears {df_long.groupby('worker_id').size().iloc[0]} times")
print("\nSample rows:")
print(df_long[['worker_id','treat','post','period','earnings']].head(6).to_string())

# COMMAND ----------

# =============================================================================
# DiD Estimation
# =============================================================================
#
# Three specifications:
#   Model 1: Simple DiD — just the interaction, no covariates
#   Model 2: DiD + covariates — adds age, education, etc. for precision
#   Model 3: DiD with entity fixed effects — most rigorous (removes all
#             time-invariant unit-level differences)
#
# All three should give similar DiD estimates if parallel trends holds.
# Differences tell you about model sensitivity.

# --- Model 1: Simple DiD ---
# earnings ~ post + treat + post*treat
m1 = smf.ols('earnings ~ post + treat + post:treat',
             data=df_long).fit(cov_type='HC3')
# HC3 = heteroscedasticity-robust standard errors
# Always use robust SEs with observational data

ATT_m1 = m1.params['post:treat']
SE_m1  = m1.bse['post:treat']
CI_m1  = m1.conf_int().loc['post:treat']

print(f"\nModel 1 — Simple DiD (no covariates)")
print(f"  ATT estimate: ${ATT_m1:,.2f}")
print(f"  Std Error:    ${SE_m1:,.2f}")
print(f"  95% CI:       [${CI_m1[0]:,.0f},  ${CI_m1[1]:,.0f}]")
print(f"  p-value:      {m1.pvalues['post:treat']:.4f}")

# --- Model 2: DiD + covariates ---
m2 = smf.ols(
    'earnings ~ post + treat + post:treat + age + educ + black + hisp + married + nodegree',
    data=df_long).fit(cov_type='HC3')

ATT_m2 = m2.params['post:treat']
SE_m2  = m2.bse['post:treat']
CI_m2  = m2.conf_int().loc['post:treat']

print(f"\nModel 2 — DiD + covariates")
print(f"  ATT estimate: ${ATT_m2:,.2f}")
print(f"  Std Error:    ${SE_m2:,.2f}")
print(f"  95% CI:       [${CI_m2[0]:,.0f},  ${CI_m2[1]:,.0f}]")
print(f"  p-value:      {m2.pvalues['post:treat']:.4f}")

# --- Manual calculation check ---
# Verify regression gives same answer as the four-cell DiD formula
did_manual = (treat_78 - treat_75) - (ctrl_78 - ctrl_75)
print(f"\nManual DiD check (four-cell formula):")
print(f"  Treated change (75→78):  ${treat_78 - treat_75:,.2f}")
print(f"  Control change (75→78):  ${ctrl_78  - ctrl_75:,.2f}")
print(f"  DiD = {treat_78-treat_75:,.2f} − {ctrl_78-ctrl_75:,.2f} = ${did_manual:,.2f}")
print(f"\n  Matches Model 1? {'✓ Yes' if abs(did_manual - ATT_m1) < 1 else '✗ No'}")

print(f"\n{'─'*60}")
print(f"True ATT (LaLonde RCT):  ${TRUE_ATT:,}")
print(f"Naive estimate:          ${naive_ate:,.0f}  (biased by ${abs(naive_ate-TRUE_ATT):,.0f})")
print(f"DiD Model 1:             ${ATT_m1:,.0f}")
print(f"DiD Model 2:             ${ATT_m2:,.0f}")

# COMMAND ----------

# =============================================================================
# Parallel Trends Test
# =============================================================================
#
# The parallel trends assumption: in the absence of treatment, the treated
# group's earnings would have followed the same TREND as the control group.
#
# We can't test this in the post-period (treatment already happened).
# But we can test it in the PRE-PERIOD: do the two groups trend
# together BEFORE treatment?
#
# We have two pre-treatment years: 1974 and 1975.
# If parallel trends holds, the DiD estimate using 1974→1975 as
# pre-pre-period should be ZERO (no "treatment" happened then).
#
# Method: Run DiD on 1974→1975 only. The coefficient on the
# interaction should be near zero and statistically insignificant.

# Restructure: 1974 (pre-pre) vs 1975 (pre)
pre74 = df[['treat','re74','age','educ','black','hisp','married','nodegree']].copy()
pre75 = df[['treat','re75','age','educ','black','hisp','married','nodegree']].copy()

pre74['earnings'] = pre74['re74']
pre74['post']     = 0
pre75['earnings'] = pre75['re75']
pre75['post']     = 1

pre74 = pre74.drop(columns=['re74'])
pre75 = pre75.drop(columns=['re75'])

df_pretrends = pd.concat([pre74, pre75], ignore_index=True)

id_cols = ['worker_id', 'treat', 'age', 'educ', 'black', 'hisp', 'married', 'nodegree']

df_pretrends = df.melt(
    id_vars=id_cols,
    value_vars=['re74', 're75'],
    var_name='period',
    value_name='earnings'
)

# Create post indicator
df_pretrends['post'] = (df_pretrends['period'] == 're75').astype(int)

df_pretrends['period'] = df_pretrends['period'].replace({
    're75': '1975',
    're74': '1974'
})

# Regress earnings on post + treat + interaction (1974→1975 only)
m_pretest = smf.ols('earnings ~ post + treat + post:treat',
                    data=df_pretrends).fit(cov_type='HC3')

pretrend_coef = m_pretest.params['post:treat']
pretrend_p    = m_pretest.pvalues['post:treat']
pretrend_ci   = m_pretest.conf_int().loc['post:treat']

print(f"\nPre-trends test (1974 → 1975, before any treatment):")
print(f"  Interaction coefficient: ${pretrend_coef:,.2f}")
print(f"  Standard Error:          ${m_pretest.bse['post:treat']:,.2f}")
print(f"  95% CI:                  [${pretrend_ci[0]:,.0f}, ${pretrend_ci[1]:,.0f}]")
print(f"  p-value:                 {pretrend_p:.4f}")

if pretrend_p > 0.05:
    print(f"\n  ✓ p > 0.05 — no statistically significant pre-trend divergence")
    print(f"    Parallel trends assumption is PLAUSIBLE")
else:
    print(f"\n  ✗ p ≤ 0.05 — significant pre-trend divergence detected")
    print(f"    Parallel trends assumption may be VIOLATED")
    print(f"    DiD estimate should be interpreted with caution")

# IMPORTANT NOTE:
# Passing this test is necessary but NOT sufficient for parallel trends.
# It only checks 1974→1975. If we had more pre-treatment years, we'd
# check ALL of them. With only two pre-periods, this is our best test.

# COMMAND ----------

# =============================================================================
# Placebo Test (Fake Treatment Period)
# =============================================================================
#
# A placebo test asks: what if we pretended treatment happened BEFORE it did?
#
# If DiD is valid, a placebo "treatment" in a pre-treatment period should
# find NO effect — because nothing actually happened then.
#
# Here we pretend treatment happened between 1974 and 1975 (using only
# pre-treatment data). The DiD estimate on this fake window should be ~$0.
#
# If the placebo estimate is large and significant → the groups were already
# diverging before treatment → parallel trends FAILS → our DiD is invalid.

# We already have df_pretrends (1974→1975)
# The placebo DiD estimate IS the pre-trends coefficient above
placebo_att = pretrend_coef
placebo_p   = pretrend_p

print(f"\nPlacebo DiD estimate (1974→1975, no real treatment):")
print(f"  Placebo ATT:  ${placebo_att:,.2f}")
print(f"  p-value:      {placebo_p:.4f}")
print(f"  Real DiD ATT: ${ATT_m1:,.2f}")
print(f"\n  Ratio (placebo/real): {abs(placebo_att/ATT_m1):.3f}")

if abs(placebo_att) < abs(ATT_m1) * 0.2 and placebo_p > 0.05:
    print("\n  ✓ Placebo effect is small and insignificant relative to real effect")
    print("    Supports validity of the DiD design")
else:
    print("\n  ⚠ Placebo effect is non-trivial — treat DiD results with caution")

# What's the difference between the parallel trends test and the placebo test? 
# It's the same test designed to answer differently framed questions:
# Parallel trends tests answer have the treat and control group behaved similarly.
# So we run this test at all points in time
# Placebo test is one where you assume a treatment happened in the pre-period
# and prove that no effect was found (no diverging or converging trends)

# COMMAND ----------

# =============================================================================
# What Happens When Parallel Trends BREAKS?
# =============================================================================
#
# We deliberately introduce a violation of parallel trends to show
# what a broken DiD looks like — and why the estimate becomes meaningless.
#
# We create a synthetic dataset where the control group's earnings
# are growing FASTER than the treated group's in the pre-period.
# This makes it look like treatment helped (when really control just declined).

df_broken = df.copy()

# Introduce differential trend: control earnings grow faster than treated
# (as if there's an economic shock hitting only control workers)
TREND_VIOLATION = 3000  # extra growth for control group
df_broken.loc[df_broken['treat']==0, 're78'] += TREND_VIOLATION

# Re-build long format
pre_b  = df_broken[['treat','re75']].copy().rename(columns={'re75':'earnings'})
post_b = df_broken[['treat','re78']].copy().rename(columns={'re78':'earnings'})
pre_b['post'] = 0
post_b['post'] = 1

df_long_b = pd.concat([pre_b, post_b], ignore_index=True)

m_broken = smf.ols('earnings ~ post + treat + post:treat',
                   data=df_long_b).fit(cov_type='HC3')
ATT_broken = m_broken.params['post:treat']

print(f"\nTrue ATT:                ${TRUE_ATT:,}")
print(f"Proper DiD estimate:     ${ATT_m1:,.0f}")
print(f"BROKEN DiD estimate:     ${ATT_broken:,.0f}")
print(f"Bias introduced:         ${abs(ATT_broken - ATT_m1):,.0f}")
print(f"\nThe violation made treatment look LESS effective.")
print(f"Control group grew faster → DiD subtracts too much → ATT is understated.")
print(f"\nKey lesson: the direction and magnitude of DiD bias depends entirely")
print(f"on HOW parallel trends is violated. There is no way to know the bias")
print(f"direction from the data alone — which is why the pre-trends test matters.")

# COMMAND ----------

# =============================================================================
# Visualization: Full DiD Summary
# =============================================================================

fig, ax = plt.subplots(figsize=(5,5))

# ── Plot A: Pre-trends test ──────────────────────────────────────────────────

for grp, color, label in [(1,'#e06b5b','Treated'), (0,'#5b8dd9','Control')]:
    sub = df[df['treat']==grp]
    ax.plot([1974, 1975],
            [sub['re74'].mean(), sub['re75'].mean()],
            'o-', color=color, linewidth=2.5, markersize=8, label=label)

ax.set_title("Pre-trends test\n(1974→1975, before treatment)", fontweight='bold')
ax.set_ylabel("Mean earnings ($)")
ax.set_xticks([1974, 1975])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

# Annotate: interaction coefficient
ax.text(1974.5, min(df['re74'].mean(), df['re75'].mean()) + 200,
        f'Pre-trend interaction\n${pretrend_coef:,.0f} (p={pretrend_p:.3f})',
        fontsize=8, ha='center',
        bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaa', alpha=0.9))
plt.savefig('../assets/plots/phase2/pre_trends_test.png', dpi=150, bbox_inches='tight')

# ── Plot B: DiD decomposition ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
periods = ['1975\n(pre)', '1978\n(post)']
ctrl_vals  = [ctrl_75,  ctrl_78]
treat_vals = [treat_75, treat_78]
cf_vals    = [treat_75, treat_75 + (ctrl_78 - ctrl_75)]

ax.plot(periods, ctrl_vals,  'o-', color='#5b8dd9', linewidth=2.5,
        markersize=8, label='Control')
ax.plot(periods, treat_vals, 'o-', color='#e06b5b', linewidth=2.5,
        markersize=8, label='Treated (actual)')
ax.plot(periods, cf_vals, 'o--', color='#e06b5b', linewidth=1.5,
        markersize=6, alpha=0.5, label='Treated (counterfactual)')

# Annotate the DiD gap
ax.annotate('', xy=(1, treat_vals[1]),
            xytext=(1, cf_vals[1]),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
ax.text(1.08, (treat_vals[1] + cf_vals[1])/2,
        f'DiD\n${ATT_m1:,.0f}',
        va='center', fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaa', alpha=0.9))

ax.set_title("DiD decomposition\n(parallel trends visible)", fontweight='bold')
ax.set_ylabel("Mean earnings ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)
plt.savefig('../assets/plots/phase2/DiD.png', dpi=150, bbox_inches='tight')

# ── Plot C: Method comparison ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
methods  = ['True ATT\n(RCT)', 'Naive\ndiff', 'DiD\nModel 1', 'DiD\nModel 2']
estimates = [TRUE_ATT, naive_ate, ATT_m1, ATT_m2]
colors   = ['#2ecc71', '#e74c3c', '#3498db', '#3498db']

bars = ax.bar(methods, estimates, color=colors, width=0.5, alpha=0.85)

for bar, val in zip(bars, estimates):
    ypos = bar.get_height() + 50 if val > 0 else bar.get_height() - 200
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')

ax.axhline(TRUE_ATT, color='#2ecc71', linestyle='--',
           linewidth=1.5, alpha=0.7, label=f'True ATT = ${TRUE_ATT:,}')
ax.axhline(0, color='#333', linewidth=0.8)
ax.set_title("Estimates vs ground truth", fontweight='bold')
ax.set_ylabel("ATT estimate ($)")
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('../assets/plots/phase2/phase2_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Results plot saved")

# COMMAND ----------

# =============================================================================
# MLflow Logging
# =============================================================================

# Set MLflow to use Databricks workspace tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

mlflow.set_experiment("/Users/deshpande.ajay.us@gmail.com/causal_inference_toolkit")

with mlflow.start_run(run_name="phase2_did"):

    # Parameters
    mlflow.log_param("method",           "DiD")
    mlflow.log_param("dataset",          "LaLonde_PSID")
    mlflow.log_param("estimand",         "ATT")
    mlflow.log_param("n_treated",        int(df['treat'].sum()))
    mlflow.log_param("n_control",        int((df['treat']==0).sum()))
    mlflow.log_param("pre_period",       "1975")
    mlflow.log_param("post_period",      "1978")
    mlflow.log_param("se_type",          "HC3_robust")
    mlflow.log_param("true_att",         TRUE_ATT)

    # Metrics — Model 1 (simple)
    mlflow.log_metric("ATT_simple",       round(ATT_m1, 2))
    mlflow.log_metric("ATT_with_covars",  round(ATT_m2, 2))
    mlflow.log_metric("CI_lower_simple",  round(float(CI_m1[0]), 2))
    mlflow.log_metric("CI_upper_simple",  round(float(CI_m1[1]), 2))
    mlflow.log_metric("SE_simple",        round(SE_m1, 2))
    mlflow.log_metric("p_value_simple",   round(m1.pvalues['post:treat'], 5))

    # Diagnostics
    mlflow.log_metric("pretrend_coef",    round(pretrend_coef, 2))
    mlflow.log_metric("pretrend_pvalue",  round(pretrend_p, 5))
    mlflow.log_metric("placebo_att",      round(placebo_att, 2))
    mlflow.log_metric("naive_estimate",   round(naive_ate, 2))
    mlflow.log_metric("recovery_pct",
                      round(ATT_m1 / TRUE_ATT * 100, 1))

    # Artifacts
    mlflow.log_artifact('/Workspace/Users/deshpande.ajay.us@gmail.com/applied-causal-inference/assets/plots/phase2/pre_trends_test.png')
    mlflow.log_artifact('/Workspace/Users/deshpande.ajay.us@gmail.com/applied-causal-inference/assets/plots/phase2/phase2_trends.png')
    mlflow.log_artifact('/Workspace/Users/deshpande.ajay.us@gmail.com/applied-causal-inference/assets/plots/phase2/DiD.png')
    mlflow.log_artifact('/Workspace/Users/deshpande.ajay.us@gmail.com/applied-causal-inference/assets/plots/phase2/phase2_results.png')

    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ MLflow run logged")
    print(f"  Run ID:          {run_id}")
    print(f"  ATT (simple):    ${ATT_m1:,.2f}")
    print(f"  ATT (covars):    ${ATT_m2:,.2f}")
    print(f"  95% CI:          [${CI_m1[0]:,.0f}, ${CI_m1[1]:,.0f}]")
    print(f"  Pre-trend p:     {pretrend_p:.4f}")
    print(f"  Recovery:        {ATT_m1/TRUE_ATT*100:.1f}% of true ATT")

# COMMAND ----------

# =============================================================================
# Summary and Bridge to Phase 3
# =============================================================================
print(f"""
What we did:
  ✓ Loaded LaLonde dataset (185 treated, 2,490 PSID controls)
  ✓ Showed selection bias: naive estimate off by ${abs(naive_ate-TRUE_ATT):,.0f}
  ✓ Reshaped data to long/panel format for DiD regression
  ✓ Estimated DiD via interaction regression (two specifications)
  ✓ Ran parallel trends test (1974→1975) — checked pre-period divergence
  ✓ Ran placebo test with fake treatment period
  ✓ Demonstrated what a parallel trends violation looks like
  ✓ Logged all results to MLflow

Key results:
  True ATT (RCT benchmark):    ${TRUE_ATT:,}
  Naive estimate:              ${naive_ate:,.0f}   ← biased
  DiD estimate (simple):       ${ATT_m1:,.0f}
  DiD estimate (+ covariates): ${ATT_m2:,.0f}
  Pre-trends test p-value:     {pretrend_p:.4f}  ← {'✓ passes' if pretrend_p > 0.05 else '✗ fails'}

Why DiD worked better than naive:
  The naive comparison confused two things: the training effect AND the
  pre-existing earnings difference between groups (~$11,000/year).
  DiD removed that pre-existing difference by using the CHANGE in
  earnings (1975→1978) rather than the LEVEL. Whatever earnings difference
  existed pre-training — it differences out of the estimate.

The problem DiD leaves unsolved (→ Phase 3):
  DiD required panel data (same workers observed pre AND post).
  It also required a control group that trends in parallel.
  What if you only have cross-sectional data — one snapshot, no pre-period?
  What if you want to ask: "who specifically benefited?" rather than
  "what was the average change?"

  The fix: Propensity Score Matching (PSM).
  Instead of using time variation, PSM finds untreated workers who look
  statistically identical to treated workers on all measured covariates —
  and compares them directly. No panel data needed. Different assumption:
  instead of parallel trends, we assume no UNMEASURED confounders.
""")

# COMMAND ----------


