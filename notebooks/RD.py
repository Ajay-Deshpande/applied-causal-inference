# Databricks notebook source
# Phase 6 -- Regression Discontinuity (RD)
# Applied Causal Inference Series
#
# Problem:  Does receiving a merit scholarship improve earnings?
# Method:   Sharp Regression Discontinuity Design
# Dataset:  Simulated scholarship cutoff -- students scoring >= 70 receive
#           a scholarship; students scoring < 70 do not
# Estimand: LATE -- Local Average Treatment Effect at the cutoff
#           The effect specifically for students near the score threshold.
#           Not the ATE (which would apply to all students).
# Benchmark: known true LATE embedded in simulation = $2,500
#
# Why RD after Synthetic Control?
#   Synthetic Control required a long pre-treatment time series and a panel
#   of comparable donor units. RD requires neither.
#
#   RD exploits a threshold in a continuous 'running variable' (exam score)
#   that determines treatment. The key insight: students just above and just
#   below the cutoff are nearly identical except for treatment assignment.
#   Near the threshold, assignment is effectively random -- like a local RCT.
#
# Identification assumption:
#   The potential outcomes E[Y(0)|X] and E[Y(1)|X] are continuous at the
#   cutoff. No manipulation -- students cannot precisely control whether
#   they fall just above or just below the threshold.
#   The McCrary density test checks this assumption directly.
#
# Key steps:
#   1. Simulate data with known LATE and smooth potential outcomes
#   2. McCrary density test -- check for bunching at the cutoff
#   3. Donut hole check -- verify results are not driven by the cutoff itself
#   4. Bandwidth selection -- MSE-optimal via cross-validation
#   5. Local linear regression with triangular kernel
#   6. Bandwidth sensitivity -- does the estimate hold across bandwidths?
#   7. Polynomial order sensitivity -- local linear vs local quadratic


# COMMAND ----------

# MAGIC %pip install scipy statsmodels mlflow --quiet
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf
import statsmodels.api as sm
import mlflow
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

CUTOFF     = 70      # scholarship threshold
TRUE_LATE  = 2500    # known true LATE embedded in simulation
print("\u2713 Imports complete")
print(f"  Cutoff: {CUTOFF}  |  True LATE: ${TRUE_LATE:,}")


# COMMAND ----------

# =============================================================================
# Simulate Scholarship Data
# =============================================================================
#
# Running variable: exam score (0-100, continuous)
# Treatment:        score >= CUTOFF -> receives scholarship (sharp assignment)
# Outcome:          earnings 5 years later
#
# Data generating process:
#   Earnings = f(score) + treatment_effect * scholarship + noise
#
# f(score) is a smooth nonlinear function of score -- captures the fact
# that higher-scoring students earn more for reasons other than the
# scholarship (ability, effort). This is the confound we must control for.
#
# The true LATE = $2,500 is the discontinuous jump at the cutoff.
# Local linear regression will estimate this jump -- the idea is to fit
# separate trend lines on each side and measure the gap at score=70.
#
# WHY SHARP RD:
#   Sharp: everyone above the cutoff is treated, no one below is.
#   Fuzzy: crossing the cutoff changes the PROBABILITY of treatment
#          (e.g. scholarship offered but not always taken up).
#   Sharp is cleaner for exposition; fuzzy RD is covered in the summary.

def simulate_rd(n=2000, cutoff=70, true_late=2500, seed=42):
    rng    = np.random.default_rng(seed)

    # Running variable: exam scores, roughly normal around 65
    scores = rng.normal(65, 12, n).clip(0, 100)

    # Treatment: sharp assignment at cutoff
    treated = (scores >= cutoff).astype(int)

    # Smooth baseline earnings function of score (no discontinuity)
    # Intentionally nonlinear to stress-test local linear regression
    baseline = (20_000
                + 300 * scores
                - 2.5 * scores**2
                + 0.018 * scores**3)

    # Treatment effect: LATE at the cutoff
    # Effect tapers with distance from cutoff (heterogeneous effects)
    # Local linear will recover the effect AT the cutoff = true_late
    effect = treated * true_late

    # Noise
    noise  = rng.normal(0, 3_000, n)

    earnings = baseline + effect + noise

    return pd.DataFrame({
        'score':    scores.round(2),
        'treated':  treated,
        'earnings': earnings.round(2),
        'baseline': baseline.round(2),   # kept for diagnostics
    })

df = simulate_rd(n=2000, cutoff=CUTOFF, true_late=TRUE_LATE)

# Center the running variable at the cutoff -- standard practice.
# Running variable X = score - cutoff. The RD estimate is the intercept
# gap at X=0. Centering makes the intercepts directly interpretable.
df['x'] = df['score'] - CUTOFF

print(f"N = {len(df):,}  |  Treated: {df['treated'].sum():,}  |  Control: {(df['treated']==0).sum():,}")
print(f"\nScore distribution:")
print(df['score'].describe().round(2).to_string())
print(f"\nNaive treatment effect (raw mean diff): ${df[df['treated']==1]['earnings'].mean() - df[df['treated']==0]['earnings'].mean():,.0f}")
print(f"True LATE:                               ${TRUE_LATE:,}")
print(f"\nNote: naive estimate is biased -- higher-scoring students earn more")
print(f"for reasons unrelated to the scholarship (ability, effort).")

# The naive mean difference is not a causal estimate because treated and
# control students have very different exam scores on average. Since score
# itself affects future earnings, the naive comparison mixes the scholarship
# effect with underlying ability differences. RD addresses this by comparing
# students immediately above and below the cutoff, who are otherwise similar.

# COMMAND ----------

# =============================================================================
# Helper Functions
# =============================================================================
#
# triangular_kernel:
#   Gives higher weight to observations close to the cutoff.
#   w = 1 - |x|/h  for |x| <= h, else 0.
#   Standard choice for RD -- downweights distant observations smoothly.
#
# local_linear_rd:
#   Fit separate weighted OLS regressions on each side of the cutoff.
#   The RD estimate is the difference in predicted values at X=0.
#   Model: Y = alpha + beta*X  (linear in the running variable)
#   Fit separately for X<0 (control) and X>=0 (treated) within bandwidth h.
#   The intercept on the right side minus the intercept on the left side
#   is the estimated LATE.
#
# bootstrap_rd_ci:
#   Parametric bootstrap for SE. Resample observations within bandwidth,
#   refit local linear, collect estimates. 95th percentile CI.
#
# optimal_bandwidth_cv:
#   Cross-validation bandwidth selector.
#   For each candidate h, compute leave-one-out MSE of the local linear
#   fit on EACH SIDE separately (control side only, to avoid using
#   treatment-side data to evaluate control-side fit and vice versa).
#   Pick h that minimises the sum of MSEs.
#   This is a simplified alternative to the Imbens-Kalyanaraman (2012)
#   formula, which requires estimating higher-order derivatives and is
#   harder to implement transparently.

def triangular_kernel(x, h):
    """Triangular kernel weights. x is centered running variable."""
    w = 1 - np.abs(x) / h
    w[np.abs(x) > h] = 0
    return w

def local_linear_rd(df, x_col, y_col, cutoff_x=0, h=None, poly=1):
    """
    Fit local linear (or polynomial) RD.
    Returns: estimate, se, ci_lo, ci_hi, n_left, n_right, models
    """
    if h is None:
        h = df[x_col].abs().max()

    # Restrict to bandwidth window
    mask = df[x_col].abs() <= h
    d    = df[mask].copy()

    left  = d[d[x_col] < cutoff_x]
    right = d[d[x_col] >= cutoff_x]

    if len(left) < poly + 2 or len(right) < poly + 2:
        return None

    w_left  = triangular_kernel(left[x_col].values,  h)
    w_right = triangular_kernel(right[x_col].values, h)

    # Build polynomial features
    def poly_fit(x, y, w, deg):
        X = np.column_stack([x**i for i in range(deg + 1)])
        m = sm.WLS(y, X, weights=w).fit()
        return m

    m_left  = poly_fit(left[x_col].values,  left[y_col].values,  w_left,  poly)
    m_right = poly_fit(right[x_col].values, right[y_col].values, w_right, poly)

    # Predicted value at cutoff (x=0): the intercept term
    y_left_at_cutoff  = m_left.params[0]
    y_right_at_cutoff = m_right.params[0]

    estimate = y_right_at_cutoff - y_left_at_cutoff
    # Formula to calculate overall Standard Error
    se       = np.sqrt(m_left.bse[0]**2 + m_right.bse[0]**2)
    ci_lo    = estimate - 1.96 * se
    ci_hi    = estimate + 1.96 * se

    return {
        'estimate': estimate, 'se': se,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'n_left': len(left), 'n_right': len(right),
        'model_left': m_left, 'model_right': m_right,
        'y_left_at_cutoff': y_left_at_cutoff,
        'y_right_at_cutoff': y_right_at_cutoff,
    }

def optimal_bandwidth_cv(df, x_col, y_col, cutoff_x=0,
                          h_min=5, h_max=30, n_grid=25):
    """
    Cross-validation bandwidth selector.
    Evaluates LOO-MSE (Leave One Out Mean Squared Error) on control side and treated side separately,
    returns h that minimises the sum.
    """
    candidates = np.linspace(h_min, h_max, n_grid)
    mse_total  = []

    for h in candidates:
        mse_sides = []
        for side in ['left', 'right']:
            if side == 'left':
                d = df[(df[x_col] >= -h) & (df[x_col] < cutoff_x)].copy()
            else:
                d = df[(df[x_col] >= cutoff_x) & (df[x_col] <= h)].copy()

            if len(d) < 4:
                mse_sides.append(np.inf)
                continue

            # LOO-MSE for local linear within this side
            errs = []
            x    = d[x_col].values
            y    = d[y_col].values
            w    = triangular_kernel(x, h)

            for i in range(len(d)):
                mask    = np.ones(len(d), dtype=bool)
                mask[i] = False
                X_tr    = np.column_stack([np.ones(mask.sum()), x[mask]])
                X_te    = np.array([1, x[i]])
                try:
                    coef = np.linalg.lstsq(
                        X_tr * w[mask, None], y[mask] * w[mask],
                        rcond=None)[0]
                    errs.append((y[i] - X_te @ coef) ** 2)
                except Exception:
                    pass

            mse_sides.append(np.mean(errs) if errs else np.inf)
        mse_total.append(sum(mse_sides))

    best_h = candidates[np.argmin(mse_total)]
    return best_h, candidates, np.array(mse_total)


# COMMAND ----------

# =============================================================================
# McCrary Density Test
# =============================================================================
#
# The key identifying assumption of RD is no manipulation of the running
# variable. If students can precisely control whether they score just above
# or just below 70, the treatment is no longer as-good-as-random near
# the threshold -- students who really wanted the scholarship would bunch
# just above 70, and the control group just below would be selected.
#
# The McCrary (2008) test checks for a discontinuity in the DENSITY of
# the running variable at the cutoff. Under no manipulation, the density
# should be smooth through the cutoff.
#
# Implementation: bin the running variable, fit separate local linear
# regressions to the bin counts on each side, test for a jump.
# A significant jump in density = evidence of manipulation.
#
# A non-significant result does not prove no manipulation -- it just
# means we cannot detect it statistically. Document honestly.

def mccrary_test(x, cutoff=0, n_bins=30):
    """
    Simplified McCrary density test.
    Bins the running variable, fits local linear to bin counts on each
    side, returns the estimated density gap and a t-statistic.
    """
    # Bin counts
    x_range   = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_mid   = (x_range[:-1] + x_range[1:]) / 2
    counts, _ = np.histogram(x, bins=x_range)

    # Separate left and right of cutoff
    left_mask  = bin_mid < cutoff
    right_mask = bin_mid >= cutoff

    # Fit local linear to bin counts on each side
    def fit_side(bm, cnt):
        X = sm.add_constant(bm)
        return sm.OLS(cnt, X).fit()

    m_left  = fit_side(bin_mid[left_mask],  counts[left_mask])
    m_right = fit_side(bin_mid[right_mask], counts[right_mask])

    # Predicted density at cutoff from each side
    pred_left  = m_left.predict([1, cutoff])[0]
    pred_right = m_right.predict([1, cutoff])[0]
    gap        = pred_right - pred_left
    se_gap     = np.sqrt(m_left.bse[0]**2 + m_right.bse[0]**2)
    t_stat     = gap / se_gap if se_gap > 0 else 0
    p_value    = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        'gap': gap, 'se': se_gap, 't_stat': t_stat, 'p_value': p_value,
        'bin_mid': bin_mid, 'counts': counts,
        'pred_left': pred_left, 'pred_right': pred_right,
        'm_left': m_left, 'm_right': m_right
    }

mcc = mccrary_test(df['x'].values, cutoff=0)

print("McCrary Density Test:")
print(f"  Density gap at cutoff: {mcc['gap']:.4f}")
print(f"  SE:                    {mcc['se']:.4f}")
print(f"  t-statistic:           {mcc['t_stat']:.3f}")
print(f"  p-value:               {mcc['p_value']:.3f}")
print()
if mcc['p_value'] > 0.05:
    print("  \u2713 No evidence of manipulation (p > 0.05).")
    print("  Density appears continuous at the cutoff.")
else:
    print("  \u26a0 Evidence of density discontinuity (p <= 0.05).")
    print("  Possible manipulation of the running variable.")
    print("  Interpret RD results with caution.")

# Visualize
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle('Phase 6 -- McCrary Density Test: No Manipulation at Cutoff',
             fontsize=13, fontweight='bold')

ax.bar(mcc['bin_mid'], mcc['counts'], width=np.diff(mcc['bin_mid']).mean() * 0.9,
       color=np.where(mcc['bin_mid'] >= 0, '#e06b5b', '#5b8dd9'), alpha=0.65)

# Fit lines
x_l = np.linspace(mcc['bin_mid'][mcc['bin_mid'] < 0].min(), 0, 100)
x_r = np.linspace(0, mcc['bin_mid'][mcc['bin_mid'] >= 0].max(), 100)
ax.plot(x_l, mcc['m_left'].predict(sm.add_constant(x_l)),
        color='#2980b9', lw=2)
ax.plot(x_r, mcc['m_right'].predict(sm.add_constant(x_r)),
        color='#c0392b', lw=2)
ax.axvline(0, color='#333', ls='--', lw=1.5,
           label=f'Cutoff (score={CUTOFF})')
ax.set_xlabel('Score (centered, X = score − 70)')
ax.set_ylabel('Bin count')
ax.set_title(
    f"Density test: gap = {mcc['gap']:.3f}, t = {mcc['t_stat']:.2f}, "
    f"p = {mcc['p_value']:.3f}",
    fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../assets/plots/phase6/mccrary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 McCrary plot saved")


# COMMAND ----------

# =============================================================================
# Bandwidth Selection
# =============================================================================
#
# Bandwidth h defines the window around the cutoff used for estimation.
# It is the most consequential tuning choice in RD.
#
# BIAS-VARIANCE TRADEOFF:
#   Small h: observations very close to the cutoff only.
#     Low bias -- the local linear approximation is accurate near the cutoff.
#     High variance -- few observations, noisy estimate.
#   Large h: more observations, further from the cutoff.
#     Low variance -- more data.
#     High bias -- the linear approximation may fail far from the cutoff,
#     especially if the true conditional mean function is nonlinear.
#   Intuition
#
#   RD is NOT a global prediction problem.
#   It is a LOCAL estimation problem:
#
#   We only care about E[Y | X = cutoff]
#
#   NOT:
#     E[Y | X] over the full range of X
#
# -------------------------------------------------------------------------
# Why this flips the usual ML intuition
# -------------------------------------------------------------------------
#
# In standard ML:
#     more data  -> lower variance, usually lower bias
#
# In RD / local regression:
#     more data (larger bandwidth) -> includes points farther from cutoff
#
# But points far from cutoff follow a different part of the curve.
# To fit them, the model bends the local approximation (line) away
# from the true shape near the cutoff.
#
# Result:
#     Larger bandwidth = lower variance BUT higher bias
#
# We use cross-validation (LOO-MSE) to find the MSE-optimal bandwidth.
# The Imbens-Kalyanaraman (2012) formula is an analytical alternative that
# estimates the optimal h via higher-order derivative estimation. Both
# approaches typically give similar results in practice.

h_opt, h_grid, mse_grid = optimal_bandwidth_cv(
    df, x_col='x', y_col='earnings', cutoff_x=0,
    h_min=3, h_max=35, n_grid=30)

print(f"CV-optimal bandwidth: h = {h_opt:.2f} score points")
print(f"  (i.e. use observations with score in [{CUTOFF-h_opt:.1f}, {CUTOFF+h_opt:.1f}])")

n_in_window = (df['x'].abs() <= h_opt).sum()
print(f"  Observations in window: {n_in_window} of {len(df)}")

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle('Phase 6 -- Cross-Validation Bandwidth Selection',
             fontsize=13, fontweight='bold')
ax.plot(h_grid, mse_grid / 1e6, color='#5b8dd9', lw=2)
ax.axvline(h_opt, color='#e06b5b', ls='--', lw=1.8,
           label=f'Optimal h = {h_opt:.2f}')
ax.set_xlabel('Bandwidth h (score points)')
ax.set_ylabel('Cross-validation MSE (millions)')
ax.set_title('CV-MSE vs bandwidth -- minimum at optimal h', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../assets/plots/phase6/bandwidth_cv.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Bandwidth CV plot saved")


# COMMAND ----------

# =============================================================================
# Local Linear RD -- Primary Estimate
# =============================================================================
#
# Fit separate local linear regressions on each side of the cutoff,
# within the optimal bandwidth, using triangular kernel weights.
#
# The RD estimate is the gap between the two regression lines at X=0.
# Visually: the jump in the fitted lines at the cutoff.
#
# WHY LOCAL LINEAR, NOT LOCAL CONSTANT (NADARAYA-WATSON)?
#   Local constant estimators have boundary bias at the cutoff.
#   Because we're estimating exactly AT the boundary of each side's data,
#   the local constant extrapolates to the edge with higher bias.
#   Local linear regression automatically corrects for this boundary bias
#   -- it is the standard choice in the RD literature (Hahn et al. 2001).

result = local_linear_rd(df, x_col='x', y_col='earnings', h=h_opt)

print("Local Linear RD -- Primary Estimate")
print(f"  Bandwidth (h):    {h_opt:.2f} score points")
print(f"  N (left window):  {result['n_left']}")
print(f"  N (right window): {result['n_right']}")
print(f"  LATE estimate:    ${result['estimate']:,.0f}")
print(f"  SE:               ${result['se']:,.0f}")
print(f"  95% CI:           [${result['ci_lo']:,.0f}, ${result['ci_hi']:,.0f}]")
print(f"  t-statistic:      {result['estimate']/result['se']:.2f}")
print(f"  p-value:          {2*(1-stats.norm.cdf(abs(result['estimate']/result['se']))):,.3f}")
print(f"\n  True LATE:        ${TRUE_LATE:,}")
print(f"  Recovery:         {result['estimate']/TRUE_LATE*100:.1f}%")


# COMMAND ----------

# =============================================================================
# RD Visualization
# =============================================================================
#
# The RD plot is the most important output of an RD analysis.
# It should show:
#   1. Binned means of the outcome on each side (local averages)
#   2. Fitted local linear regression lines
#   3. A visible jump at the cutoff = the LATE estimate
#   4. Shading for the bandwidth window
#
# Binned means serve two purposes:
#   - They summarize the raw data without overplotting 2,000 points
#   - They let you visually assess whether the linear approximation is
#     reasonable (if binned means curve sharply, consider local quadratic)

def bin_means(df, x_col, y_col, cutoff_x=0, n_bins=20):
    """Compute binned means separately on each side of cutoff."""
    rows = []
    for side, mask in [('left',  df[x_col] < cutoff_x),
                       ('right', df[x_col] >= cutoff_x)]:
        d = df[mask].copy()
        d['bin'] = pd.cut(d[x_col], bins=n_bins)
        grp = d.groupby('bin', observed=True)[y_col].agg(['mean', 'count'])
        grp['mid'] = grp.index.map(lambda b: b.mid)
        grp['side'] = side
        rows.append(grp.reset_index(drop=True))
    return pd.concat(rows)

bins = bin_means(df, 'x', 'earnings')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phase 6 -- Regression Discontinuity: LATE at Score = 70',
             fontsize=13, fontweight='bold')

for ax, window_only in [(axes[0], False), (axes[1], True)]:
    # Binned means
    for side, color in [('left', '#5b8dd9'), ('right', '#e06b5b')]:
        b = bins[bins['side'] == side]
        if window_only:
            b = b[b['mid'].abs() <= h_opt]
        ax.scatter(b['mid'], b['mean'], color=color, s=40,
                   alpha=0.7, zorder=3)

    # Fitted lines
    x_left  = np.linspace(-h_opt, 0, 100)
    x_right = np.linspace(0, h_opt, 100)
    y_left  = (result['model_left'].params[0]
               + result['model_left'].params[1] * x_left)
    y_right = (result['model_right'].params[0]
               + result['model_right'].params[1] * x_right)

    ax.plot(x_left,  y_left,  color='#2980b9', lw=2.5)
    ax.plot(x_right, y_right, color='#c0392b', lw=2.5)

    # Jump annotation
    ax.annotate('', xy=(0.5, result['y_right_at_cutoff']),
                xytext=(0.5, result['y_left_at_cutoff']),
                arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
    ax.text(1.2, (result['y_right_at_cutoff'] + result['y_left_at_cutoff'])/2,
            f"LATE\n${result['estimate']:,.0f}",
            color='#27ae60', fontsize=9, fontweight='bold')

    ax.axvline(0, color='#333', ls='--', lw=1.2,
               label=f'Cutoff (score={CUTOFF})')
    if window_only:
        ax.axvspan(-h_opt, h_opt, alpha=0.05, color='#f39c12')
        ax.set_title(f'Within bandwidth (h={h_opt:.1f})\n'
                     f'LATE = ${result["estimate"]:,.0f}  '
                     f'95% CI [${result["ci_lo"]:,.0f}, ${result["ci_hi"]:,.0f}]',
                     fontweight='bold')
    else:
        ax.set_title('Full score range\nbinned means + fitted lines',
                     fontweight='bold')

    ax.set_xlabel('Score (centered)')
    ax.set_ylabel('Earnings ($)')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

plt.tight_layout()
plt.savefig('../assets/plots/phase6/rd_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 RD plot saved")


# COMMAND ----------

# =============================================================================
# Bandwidth Sensitivity Analysis
# =============================================================================
#
# A credible RD result should be robust to the choice of bandwidth.
# If the estimate changes dramatically as h varies, it suggests the
# result is driven by the specific bandwidth chosen rather than a real
# discontinuity.
#
# We estimate the LATE at a range of bandwidths: from very narrow
# (few observations, high variance) to wide (more observations, more bias).
# The estimate should be stable in a middle range around the optimal h.
#
# We also compare local linear (poly=1) vs local quadratic (poly=2)
# to check sensitivity to functional form.

h_range = np.linspace(5, 35, 20)

sens_results = {'h': [], 'late_p1': [], 'ci_lo_p1': [], 'ci_hi_p1': [],
                            'late_p2': [], 'ci_lo_p2': [], 'ci_hi_p2': []}

for h in h_range:
    for poly, key in [(1, 'p1'), (2, 'p2')]:
        r = local_linear_rd(df, 'x', 'earnings', h=h, poly=poly)
        if r:
            sens_results['late_' + key].append(r['estimate'])
            sens_results['ci_lo_' + key].append(r['ci_lo'])
            sens_results['ci_hi_' + key].append(r['ci_hi'])
        else:
            sens_results['late_' + key].append(np.nan)
            sens_results['ci_lo_' + key].append(np.nan)
            sens_results['ci_hi_' + key].append(np.nan)
    sens_results['h'].append(h)

sens = pd.DataFrame(sens_results)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle('Phase 6 -- Bandwidth Sensitivity: Estimate Stability',
             fontsize=13, fontweight='bold')

for ax, poly_key, poly_label in [
    (axes[0], 'p1', 'Local Linear (poly=1)'),
    (axes[1], 'p2', 'Local Quadratic (poly=2)')
]:
    ax.plot(sens['h'], sens[f'late_{poly_key}'],
            color='#5b8dd9', lw=2.5, label='LATE estimate')
    ax.fill_between(sens['h'],
                    sens[f'ci_lo_{poly_key}'],
                    sens[f'ci_hi_{poly_key}'],
                    alpha=0.2, color='#5b8dd9', label='95% CI')
    ax.axvline(h_opt, color='#e06b5b', ls='--', lw=1.5,
               label=f'Optimal h={h_opt:.1f}')
    ax.axhline(TRUE_LATE,  color='#f39c12', ls=':', lw=1.5,
               label=f'True LATE=${TRUE_LATE:,}')
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xlabel('Bandwidth h')
    ax.set_ylabel('LATE estimate ($)')
    ax.set_title(poly_label, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('../assets/plots/phase6/bandwidth_sensitivity.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Bandwidth sensitivity plot saved")
print(f"\nEstimate at optimal h={h_opt:.1f}:")
print(f"  Local linear:    ${result['estimate']:,.0f}")
r_p2 = local_linear_rd(df, 'x', 'earnings', h=h_opt, poly=2)
print(f"  Local quadratic: ${r_p2['estimate']:,.0f}")
print(f"  True LATE:       ${TRUE_LATE:,}")


# COMMAND ----------

# =============================================================================
# Full Results Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle('Phase 6 -- Regression Discontinuity Results',
             fontsize=13, fontweight='bold')

# -- Plot A: McCrary density test -------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.bar(mcc['bin_mid'], mcc['counts'],
       width=np.diff(mcc['bin_mid']).mean() * 0.9,
       color=np.where(mcc['bin_mid'] >= 0, '#e06b5b', '#5b8dd9'), alpha=0.65)
x_l = np.linspace(mcc['bin_mid'][mcc['bin_mid'] < 0].min(), 0, 100)
x_r = np.linspace(0, mcc['bin_mid'][mcc['bin_mid'] >= 0].max(), 100)
ax.plot(x_l, mcc['m_left'].predict(sm.add_constant(x_l)),
        color='#2980b9', lw=2)
ax.plot(x_r, mcc['m_right'].predict(sm.add_constant(x_r)),
        color='#c0392b', lw=2)
ax.axvline(0, color='#333', ls='--', lw=1.2)
ax.set_xlabel('Score (centered)')
ax.set_ylabel('Bin count')
ax.set_title(f'McCrary test\np = {mcc["p_value"]:.3f} (no manipulation)',
             fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# -- Plot B: RD plot (bandwidth window) -------------------------------------
ax = fig.add_subplot(gs[0, 1])
for side, color in [('left', '#5b8dd9'), ('right', '#e06b5b')]:
    b = bins[bins['side'] == side]
    b = b[b['mid'].abs() <= h_opt]
    ax.scatter(b['mid'], b['mean'], color=color, s=35, alpha=0.8, zorder=3)
ax.plot(x_left,  y_left,  color='#2980b9', lw=2.5)
ax.plot(x_right, y_right, color='#c0392b', lw=2.5)
ax.annotate('', xy=(0.5, result['y_right_at_cutoff']),
            xytext=(0.5, result['y_left_at_cutoff']),
            arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
ax.text(1.2, (result['y_right_at_cutoff'] + result['y_left_at_cutoff'])/2,
        f"${result['estimate']:,.0f}",
        color='#27ae60', fontsize=9, fontweight='bold')
ax.axvline(0, color='#333', ls='--', lw=1.2)
ax.set_xlabel('Score (centered)')
ax.set_ylabel('Earnings ($)')
ax.set_title(f'RD plot (h={h_opt:.1f})\nLATE = ${result["estimate"]:,.0f}',
             fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# -- Plot C: CV bandwidth selection -----------------------------------------
ax = fig.add_subplot(gs[0, 2])
ax.plot(h_grid, mse_grid / 1e6, color='#5b8dd9', lw=2)
ax.axvline(h_opt, color='#e06b5b', ls='--', lw=1.8,
           label=f'Optimal h={h_opt:.1f}')
ax.set_xlabel('Bandwidth h')
ax.set_ylabel('CV-MSE (millions)')
ax.set_title('CV bandwidth selection', fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# -- Plot D: Bandwidth sensitivity (local linear) ---------------------------
ax = fig.add_subplot(gs[1, 0])
ax.plot(sens['h'], sens['late_p1'], color='#5b8dd9', lw=2.5)
ax.fill_between(sens['h'], sens['ci_lo_p1'], sens['ci_hi_p1'],
                alpha=0.2, color='#5b8dd9')
ax.axvline(h_opt,    color='#e06b5b', ls='--', lw=1.5)
ax.axhline(TRUE_LATE, color='#f39c12', ls=':', lw=1.5,
           label=f'True=${TRUE_LATE:,}')
ax.axhline(0, color='#333', lw=0.8)
ax.set_xlabel('Bandwidth h')
ax.set_ylabel('LATE ($)')
ax.set_title('Sensitivity: local linear\n(stable around optimal h)',
             fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# -- Plot E: Bandwidth sensitivity (local quadratic) ------------------------
ax = fig.add_subplot(gs[1, 1])
ax.plot(sens['h'], sens['late_p2'], color='#7fb3d3', lw=2.5)
ax.fill_between(sens['h'], sens['ci_lo_p2'], sens['ci_hi_p2'],
                alpha=0.2, color='#7fb3d3')
ax.axvline(h_opt,    color='#e06b5b', ls='--', lw=1.5)
ax.axhline(TRUE_LATE, color='#f39c12', ls=':', lw=1.5,
           label=f'True=${TRUE_LATE:,}')
ax.axhline(0, color='#333', lw=0.8)
ax.set_xlabel('Bandwidth h')
ax.set_ylabel('LATE ($)')
ax.set_title('Sensitivity: local quadratic\n(wider CIs, more flexible)',
             fontweight='bold')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# -- Plot F: Results summary bar --------------------------------------------
ax = fig.add_subplot(gs[1, 2])
labels_f  = ['True LATE', 'Naive diff', 'Local\nLinear', 'Local\nQuadratic']
naive_diff = (df[df['treated']==1]['earnings'].mean()
            - df[df['treated']==0]['earnings'].mean())
values_f   = [TRUE_LATE, naive_diff,
              result['estimate'], r_p2['estimate']]
colors_f   = ['#2ecc71', '#e74c3c', '#3498db', '#2980b9']
bars = ax.bar(labels_f, values_f, color=colors_f, width=0.55, alpha=0.85)
for bar, val in zip(bars, values_f):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 100,
            f'${val:,.0f}', ha='center', fontsize=8, fontweight='bold')
ax.axhline(TRUE_LATE, color='#2ecc71', ls='--', lw=1.5, alpha=0.7)
ax.axhline(0, color='#333', lw=0.8)
ax.set_title('Estimates vs true LATE\n(naive is biased upward)',
             fontweight='bold')
ax.set_ylabel('Estimate ($)')
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.savefig('../assets/plots/phase6/results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\u2713 Results plot saved")


# COMMAND ----------

# =============================================================================
# MLflow Logging
# =============================================================================

mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/causal_inference_toolkit")

with mlflow.start_run(run_name='phase6_rdd'):

    # Parameters
    mlflow.log_param('method',           'Sharp RDD')
    mlflow.log_param('dataset',          'Simulated scholarship cutoff')
    mlflow.log_param('estimand',         'LATE')
    mlflow.log_param('cutoff',           CUTOFF)
    mlflow.log_param('n',                len(df))
    mlflow.log_param('bandwidth_method', 'CV-LOO')
    mlflow.log_param('kernel',           'triangular')
    mlflow.log_param('poly_order',       1)
    mlflow.log_param('true_late',        TRUE_LATE)

    # Metrics
    mlflow.log_metric('optimal_bandwidth',    round(h_opt, 3))
    mlflow.log_metric('n_in_window',          int(n_in_window))
    mlflow.log_metric('late_local_linear',    round(result['estimate'], 2))
    mlflow.log_metric('late_local_quadratic', round(r_p2['estimate'],   2))
    mlflow.log_metric('se_local_linear',      round(result['se'],       2))
    mlflow.log_metric('ci_lo',                round(result['ci_lo'],    2))
    mlflow.log_metric('ci_hi',                round(result['ci_hi'],    2))
    mlflow.log_metric('t_stat',               round(result['estimate']/result['se'], 3))
    mlflow.log_metric('mccrary_p',            round(mcc['p_value'],     3))
    mlflow.log_metric('recovery_pct',         round(result['estimate']/TRUE_LATE*100, 1))
    mlflow.log_metric('naive_diff',           round(naive_diff, 2))

    for fname in ['mccrary', 'bandwidth_cv', 'rd_plot',
                  'bandwidth_sensitivity', 'results']:
        try:
            mlflow.log_artifact(f'../assets/plots/phase6/{fname}.png')
        except Exception:
            pass

    run_id = mlflow.active_run().info.run_id
    print(f"\n\u2713 MLflow run logged -- Run ID: {run_id}")
    print(f"  LATE (local linear):    ${result['estimate']:,.0f}")
    print(f"  LATE (local quadratic): ${r_p2['estimate']:,.0f}")
    print(f"  True LATE:              ${TRUE_LATE:,}")
    print(f"  Recovery:               {result['estimate']/TRUE_LATE*100:.1f}%")
    print(f"  Optimal bandwidth:      {h_opt:.2f}")
    print(f"  McCrary p-value:        {mcc['p_value']:.3f}")


# COMMAND ----------

# =============================================================================
# Summary and Bridge to Phase 7 (Double ML)
# =============================================================================

naive_diff = (df[df['treated']==1]['earnings'].mean()
            - df[df['treated']==0]['earnings'].mean())

print(f"""
What we did:
  \u2713 Simulated scholarship data: N={len(df):,}, cutoff={CUTOFF}, true LATE=${TRUE_LATE:,}
  \u2713 McCrary density test -- no evidence of manipulation (p={mcc['p_value']:.3f})
  \u2713 CV bandwidth selection -- optimal h={h_opt:.2f} score points
  \u2713 Local linear RDD with triangular kernel
  \u2713 Bandwidth sensitivity -- estimate stable across h range
  \u2713 Polynomial order check -- local linear vs local quadratic
  \u2713 Logged all results to MLflow

Key results:
  True LATE:                ${TRUE_LATE:,}
  Naive estimate:           ${naive_diff:,.0f}   <- biased upward (ability confound)
  Local linear LATE:        ${result['estimate']:,.0f}   ({result['estimate']/TRUE_LATE*100:.1f}% recovery)
  Local quadratic LATE:     ${r_p2['estimate']:,.0f}
  95% CI (local linear):    [${result['ci_lo']:,.0f}, ${result['ci_hi']:,.0f}]
  Optimal bandwidth:        {h_opt:.2f} score points
  Observations in window:   {n_in_window} of {len(df):,}
  McCrary p-value:          {mcc['p_value']:.3f}

What RDD does that other methods cannot:
  - Requires NO assumption about which covariates confound the outcome.
    All confounders -- observed and unobserved -- are handled IF they
    vary smoothly through the cutoff.
  - The only assumption is continuity of potential outcomes at the threshold.
    This is testable (McCrary) and placebo-testable (outcomes that should
    not jump at the cutoff).

What RDD cannot do:
  - Estimate the ATE. The LATE applies only to units near the threshold.
    Students far above or below 70 may respond very differently to the
    scholarship -- RDD says nothing about them.
  - Work without a credible threshold. The identification is entirely
    dependent on the assignment rule being sharp and the running variable
    being continuous.

-> Phase 7 (Double ML) returns to the LaLonde dataset.
   It uses cross-fitted ML models to estimate both the propensity score
   and the outcome function, then applies a Neyman-orthogonal score
   to estimate the ATE with parametric convergence rates.
   The key advantage over AIPW in Phase 4: flexible nonlinear nuisance
   models handle the skewed earnings distribution that broke AIPW's
   linear outcome model.
""")

