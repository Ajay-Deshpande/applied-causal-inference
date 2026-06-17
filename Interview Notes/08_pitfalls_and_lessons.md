# 08 — Common Pitfalls and Lessons

A catalogue of failure modes that look like reasonable analyses but produce misleading results — each entry covers what happens, how to detect it, and how to avoid it.

---

### Overlap Collapse / Near-Perfect Separation

**What it means:**
- A propensity model can predict treatment assignment almost perfectly from covariates (AUC near 1.0) — treated and control units occupy almost entirely separate regions of covariate space.
- Any method relying on comparing treated and control units (matching, weighting, residualization-based methods like DML) is left with almost no genuine common ground to compare.

**How it manifests:**
- Propensity scores cluster near 0 for almost all controls and near 1 for almost all treated units.
- Matching produces very few matched pairs, or only poor-quality matches.
- IPW weights become extreme (a few observations get enormous weights).
- DML's residual treatment variable (D̃ = D − m(X)) has very small magnitude on average — there's little variation in treatment left after accounting for X, because X almost fully determines D.

**How to detect:**
- Propensity AUC > 0.90–0.95.
- Effective sample size (ESS) after weighting is a small fraction of N.
- Mean |D̃| (residual treatment) is small relative to what it is in a comparable, better-overlapping dataset.

**How to avoid / mitigate:**
- Trimming (drop observations with extreme propensity scores) — but this changes the estimand and shouldn't be threshold-shopped (see below).
- Restrict to a more comparable control group if possible (a "closer" comparison population with better overlap, even if smaller).
- If overlap genuinely cannot be improved, report this honestly — a wide, uncertain confidence interval reflecting the true lack of identification is more useful than a falsely precise one.

---

### The Large-N Precision Trap

**What it means:**
- Standard error formulas for many estimators (including DML's sandwich SE) scale roughly as `1/√N`.
- If overlap is poor but N is very large, the `1/√N` term can shrink the SE *faster* than the weak signal (small D̃, small ESS) would suggest is warranted — producing a confidence interval that looks tight and a p-value that looks significant, despite the underlying identification being weak.

**How it manifests:**
- Two datasets (or two control groups for the same treated group) give different results: the larger one has a "significant" p-value and a point estimate further from a known benchmark; the smaller one has a non-significant p-value but a point estimate closer to the benchmark.
- A naive read concludes "the larger dataset gave a better (significant) result" — backwards.

**How to detect:**
- Always compute ESS alongside N. If ESS/N is small (e.g., single-digit percentages) for the large dataset, the apparent precision is suspect regardless of what the p-value says.
- Compare mean |D̃| (or equivalent signal-strength diagnostics) across the datasets — a "significant" result built on a much weaker signal, compensated by much larger N, is the signature of this trap.

**How to avoid:**
- Never use statistical significance alone to choose between candidate analyses/datasets — check identification diagnostics (AUC, ESS, signal strength) *first*, and only then interpret p-values, and only for analyses that pass the identification checks.
- When reporting results from a large dataset with concerning overlap, report ESS prominently alongside N — "N=16,000 (ESS=375, 2.3%)" tells a very different story than "N=16,000" alone.

---

### p-Hacking via Trimming Thresholds (and Other Robustness-Check Shopping)

**What it means:**
- Trimming (dropping observations with extreme propensity scores), choice of matching caliper, choice of bandwidth (RDD), and similar "robustness check" parameters all have a legitimate range of reasonable values — and trying multiple values, then reporting the one that crosses a significance threshold, is a form of p-hacking even if each individual choice seems defensible.

**How it manifests:**
- A result is "not significant" at the default/primary specification, but becomes "significant" at one particular trimming threshold among several tried — and that threshold is the one reported as primary, with the others relegated to (or omitted from) a robustness section.

**How to detect (when reviewing someone else's work):**
- Ask whether the reported specification was pre-specified or selected after seeing results.
- Check whether robustness checks are reported symmetrically — if multiple thresholds were tried, are *all* of them shown, with the primary one chosen on grounds independent of significance (e.g., a standard convention, or matching a prior phase's methodology)?

**How to avoid (in your own work):**
- Decide on the specification (or the *set* of specifications to report) *before* seeing how each one affects significance — or report all reasonable specifications together regardless of outcome.
- When a robustness check happens to produce a "better-looking" result, explicitly flag this and report it as a robustness check, not as the primary result — and state plainly if the primary, pre-specified result remains the one being reported.
- The correct framing for a robustness check that doesn't change the conclusion: "the result is stable across specifications, and the primary specification's conclusion holds." The correct framing for one that *does* change the conclusion: "results are sensitive to this choice, here's the range," — not silently switching the primary specification.

---

### Doubly-Robust Methods Amplifying Misspecification

**What it means:**
- "Doubly robust" (e.g., AIPW) means the estimator is *consistent* if at least one of two models (propensity, outcome) is correctly specified. It does **not** mean the estimator performs *well* whenever one model is bad — the augmentation term added by the "robust" correction can, with a badly misspecified outcome model, *add* error rather than cancel it in finite samples.

**How it manifests:**
- AIPW (or similar) produces an estimate *further* from a known benchmark than the simpler IPW estimate it's supposed to improve upon.
- Typically traceable to an outcome model that's a poor fit for the outcome's actual distribution — e.g., a linear/Gaussian model applied to a heavily right-skewed outcome with a mass at zero, where the model produces systematically biased predictions (including impossible negative predictions for a non-negative outcome).

**How to detect:**
- Check outcome model fit diagnostics (residual plots, RMSE, predictions outside the plausible range of the outcome) *before* trusting the doubly-robust estimate.
- Compare the doubly-robust estimate to the simpler component estimates (IPW alone, outcome-regression alone) — if the combined estimate is an outlier relative to both components and moves *away* from a benchmark, investigate the outcome model.

**How to avoid:**
- Match the outcome model's functional form to the outcome's actual distribution — for skewed, non-negative outcomes, prefer flexible nonparametric models (gradient boosting, etc.) over linear/Gaussian models, in both AIPW and in the nuisance models for DML.
- Don't treat "doubly robust" as a label that excuses skipping model-fit diagnostics — it reduces (does not eliminate) sensitivity to misspecification of *one* of the two models, and provides no guarantee about the magnitude of error if that model is badly wrong.

---

### False-Positive Diagnostic Tests (e.g., Density/Manipulation Tests)

**What it means:**
- Diagnostic tests themselves rely on assumptions and implementation choices — a simplified or default implementation can produce a "significant" result (e.g., p ≈ 0) that looks like it's flagging a real problem (e.g., manipulation of a running variable around an RDD cutoff), when the actual cause is a feature of the implementation (e.g., a bin-based density estimator misreading ordinary curvature in a smooth distribution near the cutoff as a "jump").

**How it manifests:**
- A diagnostic test (e.g., for manipulation/sorting around a threshold) returns a result so extreme (p = 0.000) that it would imply blatant, total manipulation — implausible given what's known about the data-generating process.
- Investigating the implementation reveals that a simplified version of the test (e.g., bin-counting rather than a proper local-polynomial density estimator) is sensitive to normal density curvature, not just discontinuities.

**How to detect:**
- Implausibly extreme test statistics (p exactly 0, or far more extreme than the apparent effect size would suggest) warrant checking the test's implementation, not just accepting the output.
- Compare against a more robust/standard implementation of the same test if available.

**How to avoid:**
- Use well-validated implementations of standard diagnostic tests where possible (established packages over from-scratch simplified versions) — but if implementing from scratch (e.g., for a learning exercise), document the simplification explicitly and interpret extreme results skeptically rather than literally.
- Sanity-check diagnostic test results against visual inspection (e.g., plot the actual density near the cutoff) — a test claiming severe manipulation should correspond to a visually obvious discontinuity in the data.

---

### Conditioning on Post-Treatment Variables / "Bad Controls"

**What it means:**
- Including a variable as a "control" in a regression or matching procedure is only correct if that variable is a *pre-treatment confounder*. Including a variable that is itself *affected by treatment* (a mediator, or any post-treatment outcome) is a "bad control" — it can introduce collider bias or absorb part of the causal effect you're trying to measure.

**How it manifests:**
- A regression that controls for a variable measured *after* treatment was assigned — even if it seems like a "reasonable covariate" — produces an estimate that's biased toward zero (if it's a mediator absorbing the effect) or biased in an unpredictable direction (if it's a collider).

**How to detect:**
- For every covariate in an adjustment set, ask: "could treatment have affected this variable?" If yes (or if its timing relative to treatment is unclear/unknown), it's a candidate bad control.

**How to avoid:**
- Only adjust for variables measured (or clearly determined) *before* treatment assignment.
- When in doubt about timing, draw the DAG explicitly (file 06) and check whether the candidate control variable has an incoming arrow from treatment — if so, it's a mediator/bad control, not a confounder.

---

### CATE / Subgroup Fishing

**What it means:**
- When a flexible CATE method (CausalForest, MetaLearners) is applied, it's tempting to scan the resulting heterogeneity for "interesting" subgroups (e.g., "the effect is huge for this specific combination of covariates!") — but with enough covariates and combinations, *some* subgroup will show a large estimated effect by chance alone, even if the true effect is homogeneous.

**How it manifests:**
- A reported finding highlights one specific, often narrow, subgroup with a striking effect size, without corresponding evidence that this subgroup was identified *before* looking at the results (pre-specified) or that the finding replicates on held-out data.

**How to detect:**
- Ask whether the highlighted subgroup was specified in advance, or "discovered" by searching the CATE function's output.
- Check whether the subgroup is large enough that the effect estimate within it has a reasonable confidence interval — very small subgroups will have very wide (and often unreported) CIs.

**How to avoid:**
- Report *overall* heterogeneity patterns (e.g., feature importances for heterogeneity, CATE distribution summary statistics) as the primary finding, and treat specific narrow-subgroup findings as hypothesis-generating rather than confirmatory — ideally validated on a separate holdout sample if a specific subgroup claim is to be made.
