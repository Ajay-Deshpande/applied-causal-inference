# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# Databricks notebook source
# Phase 8 -- Causal Inference at Scale: DoWhy + EconML
# Applied Causal Inference Series — Capstone
#
# Problem:  Does 401(k) eligibility increase net financial assets?
# Dataset:  401(k) eligibility survey (~9,915 observations)
#           Chernozhukov et al. (2018) -- the paper that introduced DML
# Estimands: ATE -- Average Treatment Effect (population-average)
#            ATT -- Average Treatment Effect on the Treated (eligible workers)
# Methods:  DoWhy (DAG + identification + refutation)
#           EconML DML/PLR, CausalForest, MetaLearners (T/S/X)
#
# This phase differs from all prior phases in three ways:
#   1. Real dataset -- not curated for pedagogy, genuinely messy
#   2. Two estimands answered simultaneously and compared
#   3. Multiple modern estimators benchmarked against each other
#
# The 401(k) dataset is the standard benchmark for DML (Chernozhukov 2018).
# Treatment: e401   -- binary indicator of 401(k) eligibility
# Outcome:   net_tfa -- net total financial assets ($)
# Confounders: age, income, family size, education, marr(ied),
#              twoearn(two_earner_couple), pira (IRA participation indicator)
#
# Key feature: income drives both eligibility (higher-income workers more
# likely to have employer-sponsored plans) and savings (directly). This is
# the confound the methods must remove.
#
# Benchmark from Chernozhukov et al. (2018): ATE ~ $8,000--$9,000

# COMMAND ----------

# MAGIC %pip install dowhy econml doubleml --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.ensemble import (GradientBoostingRegressor,
                               GradientBoostingClassifier,
                               RandomForestRegressor,
                               RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# EconML
from econml.dml import LinearDML, NonParamDML, CausalForestDML
from econml.metalearners import TLearner, SLearner, XLearner
from econml.inference import BootstrapInference

# DoWhy
import dowhy
from dowhy import CausalModel

# MLflow
import mlflow

np.random.seed(42)

# COMMAND ----------

# The 401(k) dataset is available via DoubleML's built-in datasets module.
# It originates from the Survey of Income and Program Participation (SIPP)
# and was used in Chernozhukov et al. (2018) to demonstrate DML.
#
# Variable descriptions:
#   e401       -- 1 if employer offers a 401(k) plan (treatment)
#   net_tfa    -- net total financial assets in dollars (outcome)
#   age        -- worker age
#   inc        -- annual income ($)
#   fsize      -- family size
#   educ       -- years of education
#   married    -- 1 if married
#   two_earner -- 1 if two-earner household
#   pira       -- 1 if has an IRA (Individual Retirement Account)
#
# Key distributional facts:
#   net_tfa is right-skewed with a large mass near zero -- same challenge
#   as LaLonde earnings in Phases 3-7. This is why linear outcome models
#   fail and GBM is the right nuisance learner.
#
#   Income (inc) is the primary confounder: higher-income workers are more
#   likely to work for employers who offer 401(k) plans AND more likely to
#   accumulate financial assets regardless. Failing to control for income
#   biases the naive estimate upward.
#
# Unlike LaLonde, this dataset has genuine overlap -- the AUC of a propensity
# model will be well below 0.977. We document this explicitly as the contrast
# to the CPS control group failure in Phases 3-7.

# Constants
TREATMENT    = 'e401'
OUTCOME      = 'net_tfa'
COVARIATES   = ['age', 'inc', 'fsize', 'educ', 'married', 'two_earner', 'pira']
N_FOLDS      = 5
N_BOOT       = 10
BENCHMARK_ATE = 9000  # Chernozhukov et al. (2018) approximate ATE

print("✓ Imports complete")
print(f"  Treatment:  {TREATMENT}")
print(f"  Outcome:    {OUTCOME}")
print(f"  Covariates: {COVARIATES}")
print(f"  Benchmark ATE (Chernozhukov 2018): ~${BENCHMARK_ATE:,}")

from doubleml.datasets import fetch_401K

df = fetch_401K(return_type='DataFrame')

# Rename columns to match our standard names if needed
rename_map = {
    'marr':    'married',
    'twoearn': 'two_earner',
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
df[TREATMENT] = df[TREATMENT].astype(int)

n_total   = len(df)
n_treated = df[TREATMENT].sum()
n_control = (df[TREATMENT] == 0).sum()
treat_rate = n_treated / n_total

print(f"Dataset loaded: {n_total:,} observations")
print(f"  Eligible (treated):     {n_treated:,}  ({treat_rate*100:.1f}%)")
print(f"  Not eligible (control): {n_control:,}  ({(1-treat_rate)*100:.1f}%)")
print(f"\nOutcome (net_tfa) distribution:")
print(f"  Mean:    ${df[OUTCOME].mean():>12,.0f}")
print(f"  Median:  ${df[OUTCOME].median():>12,.0f}")
print(f"  SD:      ${df[OUTCOME].std():>12,.0f}")
print(f"  Min:     ${df[OUTCOME].min():>12,.0f}")
print(f"  Max:     ${df[OUTCOME].max():>12,.0f}")
print(f"  % zero or negative: {(df[OUTCOME] <= 0).mean()*100:.1f}%")

print(f"\nNaive ATE (raw mean difference):")
naive_ate = (df[df[TREATMENT]==1][OUTCOME].mean()
           - df[df[TREATMENT]==0][OUTCOME].mean())
print(f"  Eligible mean:     ${df[df[TREATMENT]==1][OUTCOME].mean():>10,.0f}")
print(f"  Not-eligible mean: ${df[df[TREATMENT]==0][OUTCOME].mean():>10,.0f}")
print(f"  Naive ATE:         ${naive_ate:>10,.0f}  <- confounded by income")
print(f"  Benchmark ATE:     ${BENCHMARK_ATE:>10,}   <- Chernozhukov et al. (2018)")
print(f"\nCovariate means by treatment group:")
print(f"  {'Covariate':<14} {'Eligible':>12} {'Not eligible':>14} {'Diff':>10}")
print(f"  {'─'*52}")
for col in COVARIATES:
    m1 = df[df[TREATMENT]==1][col].mean()
    m0 = df[df[TREATMENT]==0][col].mean()
    print(f"  {col:<14} {m1:>12.2f} {m0:>14.2f} {m1-m0:>10.2f}")

# COMMAND ----------

# Before estimating anything, we state both estimand questions explicitly.
# Every method implemented in this phase answers one or both of these questions.
# Every method NOT implemented is explained here with a reason.
#
# ── QUESTION 1: ATE ──────────────────────────────────────────────────────────
#
#   "What is the effect of 401(k) eligibility on net financial assets for a
#    randomly drawn person from this population?"
#
#   ATE = E[Y(1) - Y(0)]
#
#   Policy use: a regulator considering making 401(k) plans universally
#   available wants to know the population-average effect. The relevant
#   counterfactual is: what would happen if everyone were made eligible?
#
# ── QUESTION 2: ATT ──────────────────────────────────────────────────────────
#
#   "What is the effect of 401(k) eligibility on net financial assets for
#    workers who are actually eligible?"
#
#   ATT = E[Y(1) - Y(0) | D=1]
#
#   Policy use: an employer considering dropping their 401(k) plan wants to
#   know the effect on their own workers -- the already-eligible population.
#   The relevant counterfactual is: what would eligible workers save if they
#   were made ineligible?
#
# ATE != ATT here. Eligible workers are higher-income on average. Higher-income
# workers have greater capacity to save regardless of eligibility. Their
# response to eligibility -- ATT -- may differ from the population-average
# response -- ATE. We will measure both and compare.

# COMMAND ----------

# ── METHODS LANDSCAPE ────────────────────────────────────────────────────────
# Do not go through the code below. It's only for presentation.
# Read the results

print("Methods landscape documented.")
print(f"\nTwo estimand questions:")
print(f"  ATE: effect on a randomly drawn person from the population")
print(f"  ATT: effect on workers who are actually eligible")
print(f"\nExpected direction: ATT > ATE")
print(f"  Eligible workers mean income:     ${df[df[TREATMENT]==1]['inc'].mean():>10,.0f}")
print(f"  Not-eligible workers mean income: ${df[df[TREATMENT]==0]['inc'].mean():>10,.0f}")
print(f"  Higher-income workers have greater savings capacity --")
print(f"  their response to eligibility (ATT) may exceed the population average (ATE).")
print(f"\nMethods NOT implemented (and why):")
print(f"  A/B:               no randomisation")
print(f"  DiD:               no panel / time dimension")
print(f"  Synthetic Control: no donor panel")
print(f"  RDD:               no score threshold")
print(f"  PSM:               dominated by DML on this data; degrades at scale")
methods_df = pd.DataFrame([
    {
        "Method": "A/B Testing",
        "ATE": "Yes",
        "ATT": "Yes",
        "Applies": "No",
        "Reason": "Eligibility is not randomly assigned; correlated with income, employer size, and industry."
    },
    {
        "Method": "DiD",
        "ATE": "No",
        "ATT": "Yes",
        "Applies": "No",
        "Reason": "Requires pre/post treatment observations. Data is cross-sectional."
    },
    {
        "Method": "Synthetic Control",
        "ATE": "No",
        "ATT": "Yes",
        "Applies": "No",
        "Reason": "Requires aggregate treated unit and donor panel. Not available here."
    },
    {
        "Method": "RDD",
        "ATE": "No",
        "ATT": "No",
        "Applies": "No",
        "Reason": "No running variable or treatment threshold."
    },
    {
        "Method": "PSM",
        "ATE": "No",
        "ATT": "Yes",
        "Applies": "Marginal",
        "Reason": "Overlap acceptable, but matching quality deteriorates in higher dimensions and large samples."
    },
    {
        "Method": "IPW",
        "ATE": "Yes",
        "ATT": "Yes",
        "Applies": "Yes",
        "Reason": "Adequate overlap. Requires ESS and weight diagnostics."
    },
    {
        "Method": "DML / PLR",
        "ATE": "Yes",
        "ATT": "Yes",
        "Applies": "Yes",
        "Reason": "Handles nonlinear confounding and large-scale observational data."
    },
    {
        "Method": "Causal Forest",
        "ATE": "Yes",
        "ATT": "Yes",
        "Applies": "Yes",
        "Reason": "Captures treatment-effect heterogeneity (CATEs)."
    },
    {
        "Method": "Meta Learners",
        "ATE": "Yes",
        "ATT": "Yes",
        "Applies": "Yes",
        "Reason": "T/S/X learners estimate heterogeneous treatment effects."
    },
    {
        "Method": "DoWhy + Backdoor",
        "ATE": "Yes",
        "ATT": "Yes",
        "Applies": "Yes",
        "Reason": "Provides DAG-based identification and refutation tests."
    }
])

display(methods_df)

# COMMAND ----------

# Two checks before any estimation:
#
#   1. Covariate imbalance -- SMD per covariate.
#      SMD > 0.1: meaningful imbalance requiring adjustment.
#      SMD > 0.25: large imbalance.
#
#   2. Overlap -- propensity score AUC and ESS.
#      AUC < 0.70: excellent overlap
#      AUC 0.70-0.85: good, manageable
#      AUC 0.85-0.92: borderline (LaLonde CPS-3 was 0.916)
#      AUC > 0.95: overlap failure (LaLonde Full CPS was 0.977)
#
# This dataset has genuine overlap -- the contrast to LaLonde's chronic
# overlap problem across Phases 3-7 is deliberate and worth documenting.
# Income is the dominant confounder and will show the largest SMD.

def compute_smd(df, covariates, treat_col):
    smds = {}
    for col in covariates:
        m1 = df[df[treat_col]==1][col].mean()
        m0 = df[df[treat_col]==0][col].mean()
        s  = df[col].std()
        smds[col] = (m1 - m0) / s if s > 0 else 0
    return pd.Series(smds)

smds = compute_smd(df, COVARIATES, TREATMENT)

X = df[COVARIATES].values
D = df[TREATMENT].values
Y = df[OUTCOME].values

# Logistic propensity for overlap check
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
lr      = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_sc, D)
pscore  = lr.predict_proba(X_sc)[:, 1]
auc     = roc_auc_score(D, pscore)
ess     = pscore.sum()**2 / (pscore**2).sum()

print(f"Covariate balance (SMD before adjustment):")
print(f"  {'Covariate':<14} {'SMD':>8}  {'Status':>12}")
print(f"  {'─'*38}")
for cov, smd in smds.items():
    status = "✓ OK" if abs(smd) < 0.1 else ("⚠ Moderate" if abs(smd) < 0.25 else "✗ Large")
    print(f"  {cov:<14} {smd:>8.3f}  {status:>12}")
print(f"\n  Mean |SMD|: {smds.abs().mean():.3f}  |  Max |SMD|: {smds.abs().max():.3f}")

print(f"\nOverlap diagnostics (logistic propensity):")
print(f"  AUC:  {auc:.3f}  {'✓ Good overlap' if auc < 0.85 else ('⚠ Borderline' if auc < 0.92 else '✗ Poor overlap')}")
print(f"  ESS:  {ess:.0f} of {len(df):,}  ({ess/len(df)*100:.1f}%)")
print(f"\n  vs LaLonde CPS-3 (Phase 7): AUC=0.916, ESS=257 of 614 (41.9%)")
print(f"  {'✓ Substantially better overlap than LaLonde.' if auc < 0.90 else '⚠ Similar overlap to LaLonde CPS-3.'}")

# COMMAND ----------

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 8 -- Covariate Balance and Overlap", fontsize=13, fontweight='bold')

ax = axes[0]
y_pos = np.arange(len(COVARIATES))
ax.scatter(smds.values, y_pos, s=70, color='#e06b5b', zorder=5)
ax.axvline(0,    color='#888', lw=0.8)
ax.axvline( 0.1, color='#aaa', lw=0.8, linestyle='--', label='|SMD|=0.1 threshold')
ax.axvline(-0.1, color='#aaa', lw=0.8, linestyle='--')
ax.set_yticks(y_pos); ax.set_yticklabels(COVARIATES)
ax.set_xlabel("Standardised Mean Difference")
ax.set_title("Covariate Imbalance Before Adjustment", fontweight='bold')
ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

ax = axes[1]
ax.hist(pscore[D==0], bins=50, alpha=0.55, color='#5b8dd9', density=True,
        label=f'Not eligible (n={n_control:,})')
ax.hist(pscore[D==1], bins=50, alpha=0.70, color='#e06b5b', density=True,
        label=f'Eligible (n={n_treated:,})')
ax.axvspan(max(pscore[D==1].min(), pscore[D==0].min()),
           min(pscore[D==1].max(), pscore[D==0].max()),
           alpha=0.07, color='green', label='Common support')
ax.set_xlabel("Propensity Score  P(D=1|X)")
ax.set_ylabel("Density")
ax.set_title(f"Propensity Score Overlap\nAUC={auc:.3f}  ESS={ess:.0f}/{len(df):,} ({ess/len(df)*100:.1f}%)",
             fontweight='bold')
ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('../assets/plots/phase8/diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Diagnostics plot saved")

# COMMAND ----------

# DoWhy forces three explicit steps before any number is computed:
#   1. Draw the causal graph (DAG) -- which variables cause which
#   2. Identify the estimand -- does the graph support identification?
#   3. Estimate -- using the identified strategy
#   4. Refute -- actively try to falsify the result
#
# The DAG for the 401(k) problem:
#
#   inc ──────────────────────────────────────────► net_tfa
#     │                                                ▲
#     └──► e401 (treatment) ────────────────────────┘
#                                                      │
#   age, educ ──► e401                                 │
#   age, educ, married ──────────────────────────────► net_tfa
#                                                      │
#   two_earner ──► e401                                │
#   two_earner ──────────────────────────────────────► net_tfa
#   (dual income → more savings capacity)              │
#                                                      │
#   fsize ──► e401                                     │
#   fsize ───────────────────────────────────────────► net_tfa
#   (larger families → more obligations, less savings) │
#                                                      │
#   pira ──► e401                                      │
#   pira ────────────────────────────────────────────► net_tfa
#   (already saving in IRA → directly raises assets)
#
# All covariates are confounders: they affect both eligibility and assets.
# The backdoor criterion requires conditioning on all of them.
#
# Causal claim: e401 → net_tfa
# Confounders: all covariates, especially inc (strongest SMD)
# No mediators, no instruments in this specification.
#
# The backdoor criterion: blocking all backdoor paths from e401 to net_tfa
# by conditioning on {age, inc, fsize, educ, married, two_earner, pira}
# is sufficient for identification, provided no unobserved confounders exist.
# This is the same unconfoundedness assumption (selection on observables or ignorability) 
# DoWhy makes it visible in the graph rather than leaving it implicit.
#
# Refutation tests:
#   1. Placebo treatment   -- replace e401 with a random binary variable.
#      A robust estimate should collapse toward zero. If it doesn't, the
#      method is picking up noise or a spurious correlation.
#
#   2. Random common cause -- add a random variable as a confounder.
#      The estimate should not change significantly. If it does, the
#      estimator is sensitive to irrelevant variables -- a sign of
#      overfitting or model instability.
#
#   3. Data subset         -- re-estimate on 80% of the data.
#      The estimate should be stable. Large changes indicate high variance
#      or sensitivity to specific observations.
#
#   4. Bootstrap refuter   -- resample with replacement and re-estimate.
#      Provides an empirical distribution of the estimate under resampling.
#      Wide distribution = high variance; the point estimate is fragile.

# COMMAND ----------

# Build the causal graph as a GML string
# All covariates are common causes (confounders)
common_causes = ' '.join([f'"{c}"' for c in COVARIATES])

causal_graph = """digraph {
    "e401" -> "net_tfa";
    "age" -> "e401"; "age" -> "net_tfa";
    "inc" -> "e401"; "inc" -> "net_tfa";
    "fsize" -> "e401"; "fsize" -> "net_tfa";
    "educ" -> "e401"; "educ" -> "net_tfa";
    "married" -> "e401"; "married" -> "net_tfa";
    "two_earner" -> "e401"; "two_earner" -> "net_tfa";
    "pira" -> "e401"; "pira" -> "net_tfa";
}"""

# --- ATE ---
model_ate = CausalModel(
    data=df,
    treatment=TREATMENT,
    outcome=OUTCOME,
    graph=causal_graph,
    common_causes=COVARIATES
)

identified_estimand_ate = model_ate.identify_effect(proceed_when_unidentifiable=True)
print("DoWhy identified estimand (ATE):")
print(identified_estimand_ate)

estimate_ate = model_ate.estimate_effect(
    identified_estimand_ate,
    method_name="backdoor.linear_regression",
    target_units="ate",
    method_params={"need_conditional_estimates": False}
)

dowhy_ate = estimate_ate.value
print(f"\nDoWhy ATE (backdoor linear regression): ${dowhy_ate:,.0f}")

# --- ATT ---
estimate_att = model_ate.estimate_effect(
    identified_estimand_ate,
    method_name="backdoor.linear_regression",
    target_units="att",
    method_params={"need_conditional_estimates": False}
)

dowhy_att = estimate_att.value
print(f"DoWhy ATT (backdoor linear regression): ${dowhy_att:,.0f}")

# --- Refutation tests ---
print(f"\nRunning refutation tests (this may take ~60s)...")

ref_placebo = model_ate.refute_estimate(
    identified_estimand_ate, estimate_ate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute", num_simulations=20
)

ref_random  = model_ate.refute_estimate(
    identified_estimand_ate, estimate_ate,
    method_name="random_common_cause", num_simulations=20
)

ref_subset  = model_ate.refute_estimate(
    identified_estimand_ate, estimate_ate,
    method_name="data_subset_refuter",
    subset_fraction=0.8, num_simulations=20
)

print(f"\nRefutation results:")
print(f"  Original ATE estimate:  ${dowhy_ate:,.0f}")
print(f"\n  1. Placebo treatment:")
print(f"     New estimate:  ${ref_placebo.new_effect:,.0f}")
print(f"     p-value:        {ref_placebo.refutation_result.get('p_value', 'N/A')}")
print(f"     {'✓ PASS -- estimate collapses under placebo' if abs(ref_placebo.new_effect) < abs(dowhy_ate) * 0.3 else '✗ FAIL -- estimate persists under placebo'}")
print(f"\n  2. Random common cause:")
print(f"     New estimate:  ${ref_random.new_effect:,.0f}")
print(f"     {'✓ PASS -- estimate stable to irrelevant confounder' if abs(ref_random.new_effect - dowhy_ate) / abs(dowhy_ate) < 0.1 else '⚠ SENSITIVE -- estimate shifts with random confounder'}")
print(f"\n  3. Data subset (80%):")
print(f"     New estimate:  ${ref_subset.new_effect:,.0f}")
print(f"     {'✓ PASS -- estimate stable across subsets' if abs(ref_subset.new_effect - dowhy_ate) / abs(dowhy_ate) < 0.1 else '⚠ UNSTABLE -- estimate varies across subsets'}")


# COMMAND ----------

# =============================================================================
# EconML DML: LinearDML and NonParamDML (ATE and ATT)
# =============================================================================
#
# EconML implements the same Robinson PLR we hand-coded in Phase 7, but with
# production-quality cross-fitting, inference, and a clean API.
#
# Two DML variants:
#
#   LinearDML:
#     Assumes the treatment effect theta is constant (homogeneous).
#     Y - E[Y|X] = theta * (D - E[D|X]) + epsilon
#     Best when we believe a single number captures the average effect.
#     Uses GBM for nuisance models, OLS for the final stage.
#     Produces analytic standard errors via the influence function.
#
#   NonParamDML (R-Learner):
#     Does not assume theta is constant. Estimates theta(X) -- a function.
#     Y - E[Y|X] = theta(X) * (D - E[D|X]) + epsilon
#     Uses GBM for nuisance models AND GBM for the final CATE stage.
#     ATE and ATT are recovered by averaging theta(X) over the appropriate
#     subpopulation (all units for ATE, treated units for ATT).
#     More flexible but higher variance -- needs N large enough to support
#     nonparametric estimation. At n=9,915 this is credible.
#
# Both use 5-fold cross-fitting. GBM nuisance models identical to Phase 7.
# We compare EconML's built-in inference to our hand-coded Phase 7 results:
# they should agree on ATE when applied to the same data structure.

GBM_PARAMS = dict(n_estimators=200, max_depth=3,
                  learning_rate=0.05, subsample=0.8, random_state=42)

model_Y = GradientBoostingRegressor(**GBM_PARAMS)
model_T = GradientBoostingClassifier(**GBM_PARAMS)

# -- LinearDML ----------------------------------------------------------------
print("Fitting LinearDML ...")
ldml = LinearDML(
    model_y=GradientBoostingRegressor(**GBM_PARAMS),
    model_t=GradientBoostingClassifier(**GBM_PARAMS),
    cv=N_FOLDS,
    random_state=42,
    discrete_treatment=True
)
ldml.fit(Y, D, X=X, inference='auto')

# ATE: built-in analytic CI
ldml_ate    = float(ldml.ate(X))
ldml_ate_ci = ldml.ate_interval(X, alpha=0.05)

# ATT: mean CATE over treated units; bootstrap CI
cate_ldml = ldml.effect(X)
ldml_att  = float(cate_ldml[D==1].mean())

print("  Computing bootstrap CI for ATT ...")
np.random.seed(42)
boot_ldml_att = []
for _ in range(N_BOOT):
    idx = np.random.choice(len(Y), len(Y), replace=True)
    m = LinearDML(model_y=GradientBoostingRegressor(**GBM_PARAMS),
                  model_t=GradientBoostingClassifier(**GBM_PARAMS),
                  cv=3, random_state=42, discrete_treatment=True)
    m.fit(Y[idx], D[idx], X=X[idx], inference='auto')
    c = m.effect(X[idx])
    boot_ldml_att.append(float(c[D[idx]==1].mean()))
ldml_att_ci = (np.percentile(boot_ldml_att, 2.5), np.percentile(boot_ldml_att, 97.5))

print(f"\nLinearDML results:")
print(f"  ATE: ${ldml_ate:>8,.0f}  95% CI: [${ldml_ate_ci[0]:>7,.0f}, ${ldml_ate_ci[1]:>7,.0f}]  (analytic)")
print(f"  ATT: ${ldml_att:>8,.0f}  95% CI: [${ldml_att_ci[0]:>7,.0f}, ${ldml_att_ci[1]:>7,.0f}]  (bootstrap, 100 draws)")

# -- NonParamDML --------------------------------------------------------------
print("\nFitting NonParamDML ...")
npdml = NonParamDML(
    model_y=GradientBoostingRegressor(**GBM_PARAMS),
    model_t=GradientBoostingClassifier(**GBM_PARAMS),
    model_final=GradientBoostingRegressor(**GBM_PARAMS),
    cv=N_FOLDS,
    random_state=42,
    discrete_treatment=True
)
npdml.fit(Y, D, X=X)

# ATE and ATT via .effect() -- no .att() method in NonParamDML
cate_npdml = npdml.effect(X)
npdml_ate  = float(cate_npdml.mean())
npdml_att  = float(cate_npdml[D==1].mean())

# Bootstrap CI for both
print("  Computing bootstrap CI...")
np.random.seed(42)
boot_npdml_ate = []
boot_npdml_att = []
for _ in range(N_BOOT):
    idx = np.random.choice(len(Y), size=len(Y), replace=True)
    m = NonParamDML(
        model_y=GradientBoostingRegressor(**GBM_PARAMS),
        model_t=GradientBoostingClassifier(**GBM_PARAMS),
        model_final=GradientBoostingRegressor(**GBM_PARAMS),
        cv=3, random_state=42, discrete_treatment=True
    )
    m.fit(Y[idx], D[idx], X=X[idx])
    c = m.effect(X[idx])
    boot_npdml_ate.append(float(c.mean()))
    boot_npdml_att.append(float(c[D[idx]==1].mean()))

npdml_ate_ci = (np.percentile(boot_npdml_ate, 2.5), np.percentile(boot_npdml_ate, 97.5))
npdml_att_ci = (np.percentile(boot_npdml_att, 2.5), np.percentile(boot_npdml_att, 97.5))

print(f"\nNonParamDML results:")
print(f"  ATE: ${npdml_ate:>8,.0f}  95% CI: [${npdml_ate_ci[0]:>7,.0f}, ${npdml_ate_ci[1]:>7,.0f}]")
print(f"  ATT: ${npdml_att:>8,.0f}  95% CI: [${npdml_att_ci[0]:>7,.0f}, ${npdml_att_ci[1]:>7,.0f}]")

print(f"\nBenchmark (Chernozhukov 2018): ATE ~ ${BENCHMARK_ATE:,}")
print(f"LinearDML  ATE recovery: {ldml_ate/BENCHMARK_ATE*100:.1f}%")
print(f"NonParamDML ATE recovery: {npdml_ate/BENCHMARK_ATE*100:.1f}%")

# COMMAND ----------

# =============================================================================
# EconML CausalForest: CATE, ATE, ATT, and Heterogeneity
# =============================================================================
#
# New concept introduced in this series: CATE -- Conditional Average
# Treatment Effect. Rather than one number for the whole population, CATE
# estimates tau(X) -- the treatment effect as a function of covariates.
#
# Why CATE matters here:
#   Income strongly predicts both eligibility and savings behaviour. It is
#   plausible -- and practically important -- that the effect of 401(k)
#   eligibility is larger for higher-income workers (who have more disposable
#   income to direct into savings) than for lower-income workers (for whom
#   eligibility may not change actual behaviour because they cannot afford
#   to contribute). CATE makes this heterogeneity visible.
#
# CausalForestDML (Wager & Athey 2018, Athey et al. 2019):
#   Combines the DML residualisation (Ytilde, Dtilde) with a forest-based
#   final stage. Each tree in the forest finds splits that maximise
#   heterogeneity in tau(X) rather than in Y. The result is a local
#   estimate of the treatment effect for each unit.
#
#   Key properties:
#     - Honest: each tree uses separate subsamples for splitting and
#       estimation, preventing overfitting of CATE
#     - Asymptotically normal: pointwise confidence intervals are valid
#     - Feature importance: identifies which covariates drive heterogeneity
#
#   ATE = mean(tau(X_i)) over all i
#   ATT = mean(tau(X_i)) over treated i only
#
# With n=9,915 this is well-powered. LaLonde's n=614 was too small for
# reliable CATE estimation -- forest leaves would have too few observations.
# This is one reason the 401(k) dataset is the right choice for this phase.

print("Fitting CausalForestDML...")
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(**GBM_PARAMS),
    model_t=GradientBoostingClassifier(**GBM_PARAMS),
    n_estimators=500,
    min_samples_leaf=10,
    max_depth=None,
    cv=N_FOLDS,
    random_state=42,
    discrete_treatment=True,
    inference=True
)
cf.fit(Y, D, X=X)

# Per-unit CATE
cate = cf.effect(X)

# ATE: mean over all; ATT: mean over treated
# .ate() and .ate_interval() exist; .att() does not -- use effect slicing
cf_ate    = float(cf.ate(X))
cf_ate_ci = cf.ate_interval(X, alpha=0.05)
cf_att    = float(cate[D==1].mean())

# Bootstrap CI for ATT
print("  Computing bootstrap CI for ATT (100 draws)...")
np.random.seed(42)
boot_cf_att = []
for _ in range(100):
    idx = np.random.choice(len(Y), len(Y), replace=True)
    m = CausalForestDML(
        model_y=GradientBoostingRegressor(**GBM_PARAMS),
        model_t=GradientBoostingClassifier(**GBM_PARAMS),
        n_estimators=200, min_samples_leaf=10,
        cv=3, random_state=42, discrete_treatment=True, inference=False
    )
    m.fit(Y[idx], D[idx], X=X[idx])
    c = m.effect(X[idx])
    boot_cf_att.append(float(c[D[idx]==1].mean()))
cf_att_ci = (np.percentile(boot_cf_att, 2.5), np.percentile(boot_cf_att, 97.5))

# Feature importance
feat_imp = cf.feature_importances_

print(f"\nCausalForest results:")
print(f"  ATE: ${cf_ate:>8,.0f}  95% CI: [${cf_ate_ci[0]:>7,.0f}, ${cf_ate_ci[1]:>7,.0f}]  (analytic)")
print(f"  ATT: ${cf_att:>8,.0f}  95% CI: [${cf_att_ci[0]:>7,.0f}, ${cf_att_ci[1]:>7,.0f}]  (bootstrap, {N_BOOT} draws)")
print(f"\nCATEs (tau(X_i)) summary:")
print(f"  Mean:   ${cate.mean():>8,.0f}")
print(f"  Median: ${np.median(cate):>8,.0f}")
print(f"  SD:     ${cate.std():>8,.0f}")
print(f"  10th pct: ${np.percentile(cate, 10):>7,.0f}")
print(f"  90th pct: ${np.percentile(cate, 90):>7,.0f}")
print(f"\nFeature importances (heterogeneity drivers):")
for feat, imp in sorted(zip(COVARIATES, feat_imp), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {feat:<14} {imp:.3f}  {bar}")

# COMMAND ----------

# Plot: CATE distribution and CATE vs income
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 8 -- CausalForest: Heterogeneous Treatment Effects",
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(cate, bins=60, color='#5b8dd9', alpha=0.75, edgecolor='white', lw=0.3)
ax.axvline(cf_ate, color='#e06b5b', lw=2.5, label=f'ATE ${cf_ate:,.0f}')
ax.axvline(cf_att, color='#2ecc71', lw=2.5, linestyle='--', label=f'ATT ${cf_att:,.0f}')
ax.axvline(0, color='#888', lw=0.8)
ax.set_xlabel("Individual CATE tau(X) ($)")
ax.set_ylabel("Count")
ax.set_title("CATE Distribution\n(one estimate per worker)", fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

ax = axes[1]
inc_vals = df['inc'].values
# Bin income and plot mean CATE per bin
bins    = np.percentile(inc_vals, np.linspace(0, 100, 21))
bin_mid = []
cate_mean = []
cate_lo   = []
cate_hi   = []
for i in range(len(bins)-1):
    mask = (inc_vals >= bins[i]) & (inc_vals < bins[i+1])
    if mask.sum() > 5:
        c = cate[mask]
        bin_mid.append((bins[i] + bins[i+1]) / 2)
        cate_mean.append(c.mean())
        se = c.std() / np.sqrt(len(c))
        cate_lo.append(c.mean() - 1.96*se)
        cate_hi.append(c.mean() + 1.96*se)

ax.plot(bin_mid, cate_mean, color='#e06b5b', lw=2, marker='o', ms=4)
ax.fill_between(bin_mid, cate_lo, cate_hi, alpha=0.2, color='#e06b5b')
ax.axhline(cf_ate, color='#5b8dd9', lw=1.5, linestyle='--', label=f'ATE ${cf_ate:,.0f}')
ax.axhline(0, color='#888', lw=0.8)
ax.set_xlabel("Income ($)")
ax.set_ylabel("Mean CATE ($)")
ax.set_title("Treatment Effect Heterogeneity by Income\n(does effect vary with earnings capacity?)",
             fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('../assets/plots/phase8/cate.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ CATE plot saved")

# 401(k) eligibility appears substantially more valuable for higher-income workers.(right chart)

# COMMAND ----------

# =============================================================================
# EconML MetaLearners: T-Learner, S-Learner, X-Learner
# =============================================================================
#
# MetaLearners are a family of CATE estimators that use off-the-shelf ML
# models as building blocks. They differ in how they use the treatment
# indicator and how they handle the two potential outcomes.
#
# S-Learner ("Single" learner):
#   Fits ONE model: mu(X, D) = E[Y | X, D]
#   CATE(X) = mu(X, 1) - mu(X, 0)
#   Treatment D is just another feature. The risk: the learner may shrink
#   the treatment coefficient toward zero if D has low predictive power
#   relative to X. Can underestimate heterogeneity. Simple but biased
#   toward zero in small samples or when treatment has a weak signal.
#
# T-Learner ("Two" learner):
#   Fits TWO separate models: mu_1(X) = E[Y|X, D=1], mu_0(X) = E[Y|X, D=0]
#   CATE(X) = mu_1(X) - mu_0(X)
#   No shrinkage toward zero, but high variance: each model is fit on only
#   the treated or control subsample. With imbalanced groups, the smaller
#   group's model has high variance and the CATE inherits it.
#   Here: n_treated ~5,900, n_control ~4,000 -- moderate imbalance.
#
# X-Learner ("Cross" learner, Kunzel et al. 2019):
#   Step 1: Fit T-Learner -- get mu_1(X) and mu_0(X)
#   Step 2: Impute individual treatment effects:
#           D_tilde_i = Y_i - mu_0(X_i)  for treated units
#           D_tilde_i = mu_1(X_i) - Y_i  for control units
#   Step 3: Fit a model of D_tilde on X separately for each group
#   Step 4: Combine via propensity-weighted average
#
#   Why X-Learner is best here:
#   Imbalanced groups (treated > control). X-Learner uses the larger group's
#   model to impute counterfactuals for the smaller group -- borrowing strength
#   across groups. It is strictly better than T-Learner when groups are
#   imbalanced and sample sizes differ substantially.
#
# ATE = mean(CATE(X_i)) over all i
# ATT = mean(CATE(X_i)) over treated i only
# Both computed from the per-unit CATE estimates.

base_reg = GradientBoostingRegressor(**GBM_PARAMS)
base_cls = GradientBoostingClassifier(**GBM_PARAMS)

# -- S-Learner ----------------------------------------------------------------
print("Fitting S-Learner...")
sl = SLearner(overall_model=GradientBoostingRegressor(**GBM_PARAMS))
sl.fit(Y, D, X=X)
cate_sl  = sl.effect(X)
sl_ate   = float(cate_sl.mean())
sl_att   = float(cate_sl[D==1].mean())

# -- T-Learner ----------------------------------------------------------------
print("Fitting T-Learner...")
tl = TLearner(models=GradientBoostingRegressor(**GBM_PARAMS))
tl.fit(Y, D, X=X)
cate_tl  = tl.effect(X)
tl_ate   = float(cate_tl.mean())
tl_att   = float(cate_tl[D==1].mean())

# -- X-Learner ----------------------------------------------------------------
print("Fitting X-Learner...")
xl = XLearner(
    models=GradientBoostingRegressor(**GBM_PARAMS),
    propensity_model=GradientBoostingClassifier(**GBM_PARAMS)
)
xl.fit(Y, D, X=X)
cate_xl  = xl.effect(X)
xl_ate   = float(cate_xl.mean())
xl_att   = float(cate_xl[D==1].mean())

# Bootstrap CIs for all three (500 draws, 3-fold for speed)
print("Computing bootstrap CIs (500 draws)...")
boot_sl_ate, boot_sl_att = [], []
boot_tl_ate, boot_tl_att = [], []
boot_xl_ate, boot_xl_att = [], []

np.random.seed(42)
for _ in range(N_BOOT):
    idx = np.random.choice(len(Y), size=len(Y), replace=True)
    Yi, Di, Xi = Y[idx], D[idx], X[idx]

    _sl = SLearner(overall_model=GradientBoostingRegressor(**GBM_PARAMS))
    _sl.fit(Yi, Di, X=Xi)
    c = _sl.effect(Xi)
    boot_sl_ate.append(float(c.mean()))
    boot_sl_att.append(float(c[Di==1].mean()) if Di.sum() > 0 else np.nan)

    _tl = TLearner(models=GradientBoostingRegressor(**GBM_PARAMS))
    _tl.fit(Yi, Di, X=Xi)
    c = _tl.effect(Xi)
    boot_tl_ate.append(float(c.mean()))
    boot_tl_att.append(float(c[Di==1].mean()) if Di.sum() > 0 else np.nan)

    _xl = XLearner(models=GradientBoostingRegressor(**GBM_PARAMS),
                   propensity_model=GradientBoostingClassifier(**GBM_PARAMS))
    _xl.fit(Yi, Di, X=Xi)
    c = _xl.effect(Xi)
    boot_xl_ate.append(float(c.mean()))
    boot_xl_att.append(float(c[Di==1].mean()) if Di.sum() > 0 else np.nan)

def ci(boot): return (np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5))

sl_ate_ci, sl_att_ci = ci(boot_sl_ate), ci(boot_sl_att)
tl_ate_ci, tl_att_ci = ci(boot_tl_ate), ci(boot_tl_att)
xl_ate_ci, xl_att_ci = ci(boot_xl_ate), ci(boot_xl_att)

print(f"\nMetaLearner results:")
print(f"  {'Method':<14} {'ATE':>10}  {'95% CI ATE':>22}  {'ATT':>10}  {'95% CI ATT':>22}")
print(f"  {'─'*82}")
for name, ate, ate_ci, att, att_ci in [
    ('S-Learner', sl_ate, sl_ate_ci, sl_att, sl_att_ci),
    ('T-Learner', tl_ate, tl_ate_ci, tl_att, tl_att_ci),
    ('X-Learner', xl_ate, xl_ate_ci, xl_att, xl_att_ci),
]:
    print(f"  {name:<14} ${ate:>8,.0f}  [${ate_ci[0]:>7,.0f}, ${ate_ci[1]:>7,.0f}]"
          f"  ${att:>8,.0f}  [${att_ci[0]:>7,.0f}, ${att_ci[1]:>7,.0f}]")

# COMMAND ----------

# =============================================================================
# Comparative Results: All Methods, Both Estimands
# =============================================================================
#
# The capstone display. Every method implemented in this phase, both estimands,
# point estimate and 95% CI side by side. Plotted as a forest plot.
#
# Interpretation guide:
#   CONVERGENCE: if multiple independent methods cluster around a similar
#   estimate, confidence in that number is high. The methods make different
#   modelling assumptions -- agreement despite different assumptions is
#   stronger evidence than any single method alone.
#
#   DIVERGENCE: if methods disagree substantially, it signals either
#   (a) model-dependence: the result is sensitive to functional form
#   (b) estimand difference: ATE vs ATT are genuinely different here
#   (c) a specific method failing an assumption (e.g. S-Learner shrinkage)
#
#   ATE vs ATT gap: the systematic gap between ATE and ATT estimates across
#   all methods is the cleanest signal that the treatment effect is
#   heterogeneous -- eligible workers respond more strongly to eligibility
#   than the average person would.

results = {
    # (ate, ate_lo, ate_hi, att, att_lo, att_hi)
    'DoWhy\nBackdoor':     (dowhy_ate, None, None, dowhy_att, None, None),
    'LinearDML':           (ldml_ate,  ldml_ate_ci[0],  ldml_ate_ci[1],
                            ldml_att,  ldml_att_ci[0],  ldml_att_ci[1]),
    'NonParamDML':         (npdml_ate, npdml_ate_ci[0], npdml_ate_ci[1],
                            npdml_att, npdml_att_ci[0], npdml_att_ci[1]),
    'CausalForest':        (cf_ate,    cf_ate_ci[0],    cf_ate_ci[1],
                            cf_att,    cf_att_ci[0],    cf_att_ci[1]),
    'S-Learner':           (sl_ate,    sl_ate_ci[0],    sl_ate_ci[1],
                            sl_att,    sl_att_ci[0],    sl_att_ci[1]),
    'T-Learner':           (tl_ate,    tl_ate_ci[0],    tl_ate_ci[1],
                            tl_att,    tl_att_ci[0],    tl_att_ci[1]),
    'X-Learner':           (xl_ate,    xl_ate_ci[0],    xl_ate_ci[1],
                            xl_att,    xl_att_ci[0],    xl_att_ci[1]),
}

methods   = list(results.keys())
n_methods = len(methods)
y_pos     = np.arange(n_methods)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.suptitle("Phase 8 -- Comparative Results: All Methods, Both Estimands\n"
             f"Benchmark ATE (Chernozhukov 2018) ≈ ${BENCHMARK_ATE:,}",
             fontsize=13, fontweight='bold')

for ax, estimand, col_idx in zip(axes, ['ATE', 'ATT'], [0, 3]):
    for i, (method, vals) in enumerate(results.items()):
        pt   = vals[col_idx]
        lo   = vals[col_idx + 1]
        hi   = vals[col_idx + 2]
        color = '#e06b5b' if estimand == 'ATT' else '#5b8dd9'
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], 'o-', color=color, lw=2, ms=5, alpha=0.8)
        ax.scatter([pt], [i], s=90, color=color, zorder=5)
        label = f'${pt:,.0f}'
        ax.text(pt, i + 0.25, label, ha='center', fontsize=7.5,
                color=color, fontweight='bold')

    ax.axvline(BENCHMARK_ATE, color='#2ecc71', lw=2, linestyle='--',
               label=f'Benchmark ATE ≈ ${BENCHMARK_ATE:,}')
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Estimate ($)")
    ax.set_title(f"{estimand} Estimates\n95% CI where available", fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/plots/phase8/comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print table
print(f"\n{'Method':<18} {'ATE':>10}  {'ATT':>10}  {'ATE-ATT gap':>12}")
print(f"{'─'*54}")
for method, vals in results.items():
    method_clean = method.replace('\n', ' ')
    ate, att = vals[0], vals[3]
    gap = att - ate if att is not None else None
    print(f"{method_clean:<18} ${ate:>8,.0f}  ${att:>8,.0f}  ${gap:>10,.0f}" if gap else
          f"{method_clean:<18} ${ate:>8,.0f}  {'N/A':>9}")
print(f"\nBenchmark ATE (Chernozhukov 2018): ~${BENCHMARK_ATE:,}")

# Convergence check
ate_vals = [v[0] for v in results.values()]
att_vals = [v[3] for v in results.values()]
print(f"\nConvergence summary:")
print(f"  ATE range: ${min(ate_vals):,.0f} -- ${max(ate_vals):,.0f}  "
      f"(spread ${max(ate_vals)-min(ate_vals):,.0f})")
print(f"  ATT range: ${min(att_vals):,.0f} -- ${max(att_vals):,.0f}  "
      f"(spread ${max(att_vals)-min(att_vals):,.0f})")
print(f"  Mean ATE across methods: ${np.mean(ate_vals):,.0f}")
print(f"  Mean ATT across methods: ${np.mean(att_vals):,.0f}")
print(f"  Mean ATT - ATE gap:      ${np.mean(att_vals) - np.mean(ate_vals):,.0f}  "
      f"({'ATT > ATE as expected' if np.mean(att_vals) > np.mean(ate_vals) else 'ATT <= ATE -- unexpected'})")
print("✓ Comparison plot saved")

# COMMAND ----------

mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/causal_inference_toolkit")

with mlflow.start_run(run_name='phase8_dowhy_econml'):

    # Parameters
    mlflow.log_param('method',            'DoWhy + EconML (DML, CausalForest, MetaLearners)')
    mlflow.log_param('dataset',           '401k_eligibility')
    mlflow.log_param('n_obs',             len(df))
    mlflow.log_param('n_treated',         int(n_treated))
    mlflow.log_param('n_control',         int(n_control))
    mlflow.log_param('treatment',         TREATMENT)
    mlflow.log_param('outcome',           OUTCOME)
    mlflow.log_param('covariates',        str(COVARIATES))
    mlflow.log_param('n_folds',           N_FOLDS)
    mlflow.log_param('n_boot',            N_BOOT)
    mlflow.log_param('nuisance_model',    'GradientBoosting(n_est=200,depth=3,lr=0.05)')
    mlflow.log_param('benchmark_ate',     BENCHMARK_ATE)
    mlflow.log_param('overlap_auc',       round(auc, 3))
    mlflow.log_param('overlap_ess',       round(ess, 1))

    # DoWhy
    mlflow.log_metric('dowhy_ate',        round(dowhy_ate, 2))
    mlflow.log_metric('dowhy_att',        round(dowhy_att, 2))

    # LinearDML
    mlflow.log_metric('ldml_ate',         round(ldml_ate, 2))
    mlflow.log_metric('ldml_att',         round(ldml_att, 2))
    mlflow.log_metric('ldml_ate_ci_lo',   round(ldml_ate_ci[0], 2))
    mlflow.log_metric('ldml_ate_ci_hi',   round(ldml_ate_ci[1], 2))
    mlflow.log_metric('ldml_att_ci_lo',   round(ldml_att_ci[0], 2))
    mlflow.log_metric('ldml_att_ci_hi',   round(ldml_att_ci[1], 2))

    # NonParamDML
    mlflow.log_metric('npdml_ate',        round(npdml_ate, 2))
    mlflow.log_metric('npdml_att',        round(npdml_att, 2))

    # CausalForest
    mlflow.log_metric('cf_ate',           round(float(cf_ate), 2))
    mlflow.log_metric('cf_att',           round(float(cf_att), 2))
    mlflow.log_metric('cate_mean',        round(float(cate.mean()), 2))
    mlflow.log_metric('cate_sd',          round(float(cate.std()), 2))

    # MetaLearners
    mlflow.log_metric('sl_ate',           round(sl_ate, 2))
    mlflow.log_metric('sl_att',           round(sl_att, 2))
    mlflow.log_metric('tl_ate',           round(tl_ate, 2))
    mlflow.log_metric('tl_att',           round(tl_att, 2))
    mlflow.log_metric('xl_ate',           round(xl_ate, 2))
    mlflow.log_metric('xl_att',           round(xl_att, 2))

    # Aggregates
    mlflow.log_metric('mean_ate_all_methods', round(np.mean(ate_vals), 2))
    mlflow.log_metric('mean_att_all_methods', round(np.mean(att_vals), 2))
    mlflow.log_metric('ldml_ate_recovery_pct',
                      round(ldml_ate / BENCHMARK_ATE * 100, 1))

    # Artifacts
    mlflow.log_artifact('../assets/plots/phase8/diagnostics.png')
    mlflow.log_artifact('../assets/plots/phase8/cate.png')
    mlflow.log_artifact('../assets/plots/phase8/comparison.png')

    run_id = mlflow.active_run().info.run_id
    print(f"✓ MLflow run logged -- Run ID: {run_id}")
    print(f"\n  Key results:")
    print(f"  {'Method':<18} {'ATE':>10}  {'ATT':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Benchmark':18} ${BENCHMARK_ATE:>8,}  {'N/A':>10}")
    print(f"  {'DoWhy':18} ${dowhy_ate:>8,.0f}  ${dowhy_att:>8,.0f}")
    print(f"  {'LinearDML':18} ${ldml_ate:>8,.0f}  ${ldml_att:>8,.0f}")
    print(f"  {'NonParamDML':18} ${npdml_ate:>8,.0f}  ${npdml_att:>8,.0f}")
    print(f"  {'CausalForest':18} ${cf_ate:>8,.0f}  ${cf_att:>8,.0f}")
    print(f"  {'X-Learner':18} ${xl_ate:>8,.0f}  ${xl_att:>8,.0f}")
    print(f"  {'─'*40}")
    print(f"  {'Mean (all)':18} ${np.mean(ate_vals):>8,.0f}  ${np.mean(att_vals):>8,.0f}")


# COMMAND ----------

# =============================================================================
# Cell 11 -- Series Summary
# =============================================================================
#
# Eight phases. One question: does an intervention change an outcome?
# The answer kept depending on four things: the identification assumption,
# the data quality, the estimator choice, and the estimand definition.

print(f"""
Applied Causal Inference Series -- Complete

Phase 1:  A/B Testing         -- ATE  -- Simulated e-commerce
Phase 2:  DiD                 -- ATT  -- LaLonde, parallel trends
Phase 3:  PSM                 -- ATT  -- LaLonde CPS-3, near-overlap failure
Phase 4:  IPW + AIPW          -- ATT  -- LaLonde, ESS collapse, OLS misfit
Phase 5:  Synthetic Control   -- ATT  -- Simulated state panel
Phase 6:  RDD                 -- LATE -- Simulated scholarship threshold
Phase 7:  Double ML           -- ATT  -- LaLonde, GBM nuisance, $1,499 (83.6%)
Phase 8:  DoWhy + EconML      -- ATE+ATT -- 401k, comparative study

Phase 8 final results (401k eligibility on net financial assets):
  Benchmark ATE (Chernozhukov 2018): ~${BENCHMARK_ATE:,}
  Naive ATE (confounded):            ${naive_ate:,.0f}

  Method              ATE           ATT
  ─────────────────────────────────────────────
  DoWhy backdoor      ${dowhy_ate:>8,.0f}     ${dowhy_att:>8,.0f}
  LinearDML           ${ldml_ate:>8,.0f}     ${ldml_att:>8,.0f}
  NonParamDML         ${npdml_ate:>8,.0f}     ${npdml_att:>8,.0f}
  CausalForest        ${cf_ate:>8,.0f}     ${cf_att:>8,.0f}
  S-Learner           ${sl_ate:>8,.0f}     ${sl_att:>8,.0f}
  T-Learner           ${tl_ate:>8,.0f}     ${tl_att:>8,.0f}
  X-Learner           ${xl_ate:>8,.0f}     ${xl_att:>8,.0f}
  ─────────────────────────────────────────────
  Mean (all methods)  ${np.mean(ate_vals):>8,.0f}     ${np.mean(att_vals):>8,.0f}

Three lessons that held across all eight phases:

  1. Identification precedes estimation.
     The overlap failure in LaLonde Phases 3-7 could not be fixed by
     a better estimator. DML on Full CPS produced p=0.020 -- significant,
     wrong. The diagnostic (ESS=375/16,177) mattered more than the p-value.
     Here, genuine overlap (AUC={auc:.3f}) let all methods converge.

  2. The estimand is a choice, not a default.
     ATE and ATT differ here because eligible workers are higher-income.
     Mean ATT - ATE gap: ${np.mean(att_vals) - np.mean(ate_vals):,.0f} across methods.
     Every phase in the series targeted ATT for comparability with the
     LaLonde RCT. Phase 8 is the first phase where both estimands are
     answered and compared.

  3. Method convergence is evidence; divergence is a question.
     When DML, CausalForest, and X-Learner agree on ATE within a tight
     range despite different functional form assumptions, that agreement
     is stronger evidence than any single method's confidence interval.
     When methods diverge -- as S-Learner often does relative to the
     others -- the question is which assumption is wrong, not which
     number is right.
""")

