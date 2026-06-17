# 07 — Diagnostics

Diagnostics don't estimate causal effects — they tell you whether the *conditions required* for a chosen method to work are actually present in the data. Running diagnostics before (and after) estimation is what separates "I ran a model" from "I have a defensible causal estimate."

---

### Standardized Mean Difference (SMD) / Love Plot

**What it means:**
- Measures how imbalanced a covariate is between treated and control groups, in units of standard deviations — making imbalance comparable *across* covariates with different scales (e.g., comparing imbalance in "age" to imbalance in "income").

**Formula:**

  `SMD = (X̄₁ − X̄₀) / S`

- **X̄₁**: mean of the covariate among treated units.
- **X̄₀**: mean of the covariate among control units.
- **S**: the covariate's standard deviation (often the pooled SD across both groups, or sometimes just the control-group SD — conventions vary).
- Result is unitless — an SMD of 0.5 means the group means differ by half a standard deviation, regardless of whether the covariate is measured in years, dollars, or counts.

**Interpretation thresholds (common conventions):**
- |SMD| < 0.1 — covariate is well balanced; adjustment likely not critical for this variable.
- 0.1 ≤ |SMD| < 0.25 — moderate imbalance; adjustment is meaningfully helpful.
- |SMD| ≥ 0.25 — large imbalance; this covariate is likely a major confounder requiring careful adjustment.

**Love plot:**
- A horizontal dot plot showing SMD for each covariate, often with two sets of points — "before adjustment" and "after adjustment" (e.g., after IPW reweighting or matching) — to visually confirm that adjustment actually reduced imbalance. Vertical reference lines at ±0.1 mark the "balanced" zone.

**When to use:**
- Before estimation, to identify which covariates are the biggest sources of imbalance (and therefore the most important to adjust for correctly).
- After estimation (for matching/weighting methods), to confirm the adjustment achieved its goal — if post-adjustment SMDs are still large, the method hasn't actually balanced the groups, regardless of what the point estimate says.

---

### Propensity Score AUC (Area Under the ROC Curve)

**What it means:**
- Fit a model predicting treatment assignment D from covariates X (e.g., logistic regression, GBM), then compute the AUC of that model — how well covariates alone *distinguish* treated from control units.
- AUC = 0.5 means covariates carry no information about treatment assignment (as good as a coin flip) — this is what you'd expect under true randomization.
- AUC = 1.0 means covariates perfectly separate the groups — every treated unit is distinguishable from every control unit based on X alone.

**Interpretation thresholds (rules of thumb, not hard cutoffs):**
- AUC < 0.70 — good overlap; treated and control groups occupy substantially overlapping regions of covariate space.
- AUC 0.70–0.85 — moderate confounding by observables; manageable with standard adjustment methods.
- AUC 0.85–0.92 — borderline; overlap is becoming thin, adjustment methods may have elevated variance, worth checking ESS.
- AUC > 0.95 — likely overlap failure (near-perfect separation); covariates almost fully determine treatment assignment, meaning there's very little "common ground" left to compare treated vs. control.

**When to use:**
- As a first diagnostic on *any* observational dataset before choosing a method — high AUC is an early warning that matching/weighting-based methods will struggle, regardless of which specific method or software is used.
- To *compare* datasets or subpopulations — e.g., "Full sample has AUC=0.97 (overlap failure) but a trimmed subsample has AUC=0.92 (borderline but usable)."

**When not to use / limitations:**
- AUC alone doesn't tell you *where* the overlap problem is (which covariate regions lack common support) — pair with propensity score histograms by group, and with SMD per covariate to identify which variables are driving the separation.
- A low AUC is reassuring about *observed* confounders but says nothing about unobserved ones — overlap and unconfoundedness are separate assumptions.

---

### Effective Sample Size (ESS)

**What it means:**
- After reweighting (IPW, overlap weights, or any propensity-based weighting), some observations get very large weights and others get very small weights. ESS answers: "given this distribution of weights, how many *unweighted, equal-contribution* observations is this effectively equivalent to?"

**Formula:**

  `ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²`

- **wᵢ**: the weight assigned to observation i (e.g., the propensity score e(Xᵢ) for ATT weighting).
- If all weights are equal, ESS = N (no loss). If weights are highly unequal — a few observations have huge weights and most have near-zero weights — ESS << N, meaning the analysis is effectively being driven by a small fraction of the data even though N observations are nominally "in" the dataset.

**Interpretation:**
- ESS / N close to 1 — weighting barely changed the effective sample; minimal overlap concerns.
- ESS / N very small (e.g., 2–5%) — the overwhelming majority of the nominal sample is contributing almost nothing to the estimate; standard errors that scale with N (not ESS) can be deeply misleading here.

**When to use:**
- Whenever any propensity-based weighting is used — ESS is the single number that most directly answers "is this estimate actually supported by enough data, or is it really resting on a handful of observations?"
- As the key diagnostic for the "large-N precision trap" (see file 08) — a dataset can have enormous N and still have tiny ESS after weighting, and standard error formulas based on N (not ESS) will understate the true uncertainty.

---

### Common Support / Overlap Histograms

**What it means:**
- A direct visualization: plot the distribution of propensity scores (or of key covariates) separately for treated and control groups, overlaid.
- "Common support" is the region where both distributions have meaningful density — outside this region, one group has observations and the other doesn't, meaning there's no empirical counterfactual for those units.

**When to use:**
- Alongside AUC and ESS — AUC and ESS are summary numbers, but the histogram shows *where* the overlap problem is (e.g., "controls are concentrated near e(X)=0, treated units are spread across the full range — the mismatch is specifically among low-propensity controls").
- To decide on a trimming threshold if trimming is used — visually identify the region where one group's density drops to near-zero.

---

### Cross-Fitting (Recap)

**What it means:**
- Already covered in file 05 as part of DML/CausalForest — included here because it's fundamentally a *diagnostic-adjacent* practice: it exists specifically to prevent overfitting-induced bias in nuisance models from contaminating the causal estimate.

**When to use:**
- Any time nuisance functions (propensity models, outcome models) are estimated with flexible ML models that are capable of overfitting — essentially always, when using GBM/random forest/neural net nuisance models.

**When not to use / limitations:**
- With very small N, even cross-fitting may not be enough — each fold has even less data, and nuisance models may still be unstable. There's a floor below which flexible nuisance models plus cross-fitting underperform simpler (e.g., linear) nuisance models with no cross-fitting needed.

---

### Analytic vs. Bootstrap Confidence Intervals

**What it means:**
- **Analytic CI**: derived from a closed-form (or asymptotic) formula for the estimator's variance — e.g., the sandwich SE formula for DML, or the asymptotic variance formula for CausalForest's GRF-based estimates.
- **Bootstrap CI**: resample the data with replacement many times, recompute the estimate each time, and use the empirical distribution of those estimates (e.g., the 2.5th and 97.5th percentiles) as the CI.

**When they can disagree:**
- Analytic formulas rely on asymptotic approximations and specific variance decompositions — for some estimators (notably tree/forest-based methods like CausalForest), the analytic variance for an *aggregate* quantity (e.g., ATE = average of many leaf-level τ̂(x)) can be much larger than the empirical variability seen via bootstrap, because the analytic formula may not fully account for how leaf-level errors partially cancel when averaged.
- Bootstrap CIs can themselves be unstable with small N or computationally expensive with complex models (each bootstrap iteration re-fits the entire model).

**When to use which:**
- Use the analytic CI when it's available and the estimator's asymptotic theory is well-established for the sample size at hand (e.g., DML's sandwich SE is generally reliable).
- Use bootstrap as a cross-check whenever an analytic CI looks surprising (implausibly wide, implausibly narrow, or crossing zero when the point estimate is far from zero and consistent across methods) — and report both if they disagree, rather than picking whichever looks better.

**When not to use / limitations:**
- Don't report only the CI that "looks better" (narrower / doesn't cross zero) without disclosing that the other type of CI gave a different answer — this is a subtle form of result-shopping, and the discrepancy itself is informative about estimator stability.
