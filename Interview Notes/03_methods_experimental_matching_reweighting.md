# 03 — Methods: Experimental, Matching, and Reweighting

---

### RCT / A-B Testing (Randomized Controlled Trial)

**What it means:** Treatment is assigned to units via a random mechanism (coin flip, random number) independent of any covariates. Because assignment is random, treated and control groups are — in expectation — identical on every observed *and unobserved* characteristic. The simple difference in mean outcomes E[Y|D=1] − E[Y|D=0] is an unbiased estimate of the ATE.

**Assumptions:** Randomization (by construction); SUTVA (no interference/spillovers between arms, well-defined treatment); sufficient sample size for the randomization to actually balance groups in realized data (randomization guarantees balance *in expectation*, not in every finite sample — check balance tables even in RCTs).

**Advantages:** The only method that addresses *unobserved* confounding by design — no need to assume unconfoundedness, because randomization makes it true. Simplest possible estimator and inference. The benchmark every observational method is implicitly compared against.

**Disadvantages:** Often infeasible — ethically (can't randomly deny a beneficial treatment), practically (can't randomize macro policy), or in terms of cost/time. Internal validity is high but external validity can be limited (a trial population may not represent the deployment population). Non-compliance turns the clean ATE into an ITT, requiring further assumptions (e.g., IV/LATE) to recover the effect of actual treatment receipt.

---

### Propensity Score Matching (PSM)

**What it means:** Estimate the propensity score e(X) = P(D=1|X) for every unit (typically via logistic regression or another classifier). For each treated unit, find one or more control units with a similar propensity score, and compare outcomes within these matched pairs/groups. The ATT is the average outcome difference across matched pairs.

Common variants:
- **Nearest-neighbor matching** — match each treated unit to the control with the closest propensity score.
- **Caliper matching** — only match if the propensity score difference is below a threshold; unmatched units are dropped.
- **Exact matching** — match only on identical covariate values (feasible only for few, discrete covariates).
- **Coarsened Exact Matching (CEM)** — discretize continuous covariates into bins, then exact-match on the bins.

**Assumptions:** Unconfoundedness given X; positivity/overlap (matches must exist); correct propensity model specification (matching quality depends on e(X) being estimated well, though the matching itself is somewhat robust to propensity model misspecification compared to outcome-regression approaches).

**Advantages:** Conceptually transparent — "compare similar units" is intuitive to non-technical stakeholders. Non-parametric in the outcome — doesn't require modeling E[Y|X,D], only the propensity score. Produces a directly interpretable matched sample that can be inspected.

**Disadvantages:** Degrades badly in high-dimensional covariate space ("curse of dimensionality") — with many continuous covariates, finding genuinely close matches becomes difficult, and match quality falls even as N grows. Discards data — unmatched units are dropped, which can substantially shrink the effective sample and change the estimand (you're now estimating the effect for the *matchable* subpopulation, which may differ from the full treated population). Computationally expensive at scale — pairwise distance comparisons don't scale well to very large N. Sensitive to which matching algorithm/caliper is chosen — different reasonable choices can give meaningfully different estimates (researcher degrees of freedom).

---

### Inverse Probability Weighting (IPW)

**What it means:** Reweight observations so that the weighted distribution of covariates is balanced between treated and control groups, mimicking what randomization would have produced. For ATE: weight treated units by 1/e(X) and control units by 1/(1−e(X)). For ATT: weight treated units by 1, and control units by e(X)/(1−e(X)) (this up-weights controls who "look like" treated units).

θ_ATE = E[ D·Y/e(X) ] − E[ (1−D)·Y/(1−e(X)) ]

**Stabilized weights:** Multiply the basic weights by the marginal probability of treatment, P(D=d), to reduce variance — the weights then have mean ≈ 1 rather than potentially very large or small values.

**Trimming:** Drop (or cap) observations with extreme propensity scores (e.g., e(X) < 0.05 or > 0.95) before computing weights — these observations would otherwise receive enormous weight and dominate the variance of the estimate.

**Assumptions:** Unconfoundedness given X; positivity (weights are undefined/infinite if e(X)=0 or 1 for any unit with that covariate profile); correct propensity model specification (IPW relies entirely on e(X) being correctly modeled — there's no outcome model to fall back on).

**Advantages:** Uses the full dataset (no matching/dropping, aside from trimming) — generally more efficient than matching when overlap is reasonable. Conceptually connects directly to "what would this look like if treatment had been randomly assigned, after reweighting."

**Disadvantages:** Highly sensitive to extreme propensity scores — a handful of observations with e(X) near 0 or 1 can receive enormous weights and dominate the estimate's variance (effective sample size collapse). Entirely dependent on correct propensity model specification — if e(X) is wrong, there's no second line of defense (unlike AIPW). Trimming improves stability but changes the estimand — you're now estimating the effect for the population *within the trimmed propensity range*, and aggressive trimming-threshold search risks p-hacking (searching for a threshold that produces a desired significance level).

---

### AIPW — Augmented Inverse Probability Weighting (Doubly Robust)

**What it means:** Combines an outcome regression model with IPW. The estimator adds an "augmentation" term that corrects the IPW estimate using predicted outcomes:

θ_AIPW = E[ μ₁(X) − μ₀(X) ] + E[ D·(Y−μ₁(X))/e(X) ] − E[ (1−D)·(Y−μ₀(X))/(1−e(X)) ]

where μ_d(X) = E[Y|X,D=d] is the outcome model. The key property — "double robustness" — is that the estimator is consistent if *either* the propensity model e(X) *or* the outcome model μ(X) is correctly specified (not necessarily both).

**Assumptions:** Unconfoundedness given X; positivity; correct specification of *at least one* of {e(X), μ(X)}.

**Advantages:** Theoretically the best of both worlds — two chances to get the model right instead of one. When both models are reasonably good, AIPW is typically more efficient (lower variance) than IPW alone.

**Disadvantages:** "Double robustness" is a guarantee about *consistency under correct specification of one model* — it is not a guarantee that AIPW will outperform either component method when *both* models are misspecified, or even when one is badly misspecified. In practice, a poorly-specified outcome model (e.g., linear regression on a heavily skewed outcome) can inject its own bias into the augmentation term, and this bias can *amplify* rather than cancel — producing an AIPW estimate that is *worse* than plain IPW. The "augmentation" is only a correction if the thing being corrected toward is itself reasonable; a bad μ(X) is not a neutral addition.

---

### Overlap Weights

**What it means:** A reweighting scheme designed to directly target the population with the best covariate overlap — each unit is weighted by the probability of being in the *opposite* group: treated units get weight (1−e(X)), control units get weight e(X). This weighting scheme bounds all weights between 0 and 1 (unlike IPW, where weights can be arbitrarily large), and naturally down-weights units in regions of poor overlap rather than requiring an arbitrary trimming threshold.

**Assumptions:** Same core assumptions as IPW (unconfoundedness, correct propensity specification) — but positivity violations are handled more gracefully because weights are bounded.

**Advantages:** Avoids the extreme-weight problem of IPW without needing to choose a trimming threshold — the weighting function itself smoothly handles poor-overlap regions. The implied estimand ("average treatment effect in the overlap population," ATO) is often a more honest description of what the data can actually support.

**Disadvantages:** Changes the estimand — ATO is not ATE or ATT, and that distinction needs to be communicated to stakeholders who may expect a population-level number. Less commonly implemented in standard software compared to IPW/PSM, so requires more custom implementation.
