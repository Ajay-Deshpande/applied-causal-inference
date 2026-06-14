# Applied Causal Inference

A practitioner's series implementing causal inference methods from scratch —
on real datasets, in Databricks, with MLflow tracking.

**Read the series:** https://Ajay-Deshpande.github.io/applied-causal-inference/

**Medium:** https://ajay-deshpande.medium.com/why-most-data-science-doesnt-answer-the-question-you-re-asking-508aad563418

## Notebooks

| Phase | Method | Dataset | Estimand | Status |
|-------|--------|---------|----------|--------|
| 1 | A/B Testing / RCT | Simulated e-commerce | ATE | ✓ Live |
| 2 | Difference-in-Differences | LaLonde (1986) | ATT | ✓ Live |
| 3 | Propensity Score Matching | LaLonde + CPS controls | ATT | ✓ Live |
| 4 | IPW + AIPW | LaLonde + CPS-3 | ATT | ✓ Live |
| 5 | Synthetic Control | Simulated state panel | ATT | ✓ Live |
| 6 | Regression Discontinuity | Simulated scholarship | LATE | ✓ Live |
| 7 | Double Machine Learning | LaLonde + CPS-3 | ATT | ✓ Live |
| 8 | DoWhy + EconML — Comparative Capstone | 401(k) eligibility | ATE + ATT | ✓ Live |

The series is complete. Phase 8 closes it out as a capstone: one real dataset,
both ATE and ATT estimated simultaneously, and seven methods (DoWhy backdoor,
EconML LinearDML, NonParamDML, CausalForest, S/T/X-Learners) compared side by
side against a published benchmark.

## What each phase covers

**Phase 1 — A/B Testing.** The gold standard: randomisation removes
confounding by design. Establishes the framing — potential outcomes,
estimands, and why everything after this phase exists to approximate what
randomisation gives for free.

**Phase 2 — Difference-in-Differences.** LaLonde (1986) job training data
with PSID controls. Uses the time dimension as the control: parallel trends
assumption, pre-trends test, and a documented case of how the assumption
breaks.

**Phase 3 — Propensity Score Matching.** LaLonde with full CPS controls
(15,992) and a trimmed CPS-3 subset (429). The full CPS group has near-zero
overlap (AUC ≈ 1.0) — PSM is inapplicable there. The trimmed group drops 55%
of treated units and still produces a CI crossing zero. An honest failure,
documented as such.

**Phase 4 — IPW and AIPW.** Same data as Phase 3. Trimmed IPW recovers ATT ≈
$1,212 (68% of the $1,794 RCT benchmark). AIPW — theoretically doubly robust —
falls to $476 because its OLS outcome model misfits right-skewed earnings.
The augmentation amplifies the misspecification instead of correcting it.

**Phase 5 — Synthetic Control.** A simulated state-level panel. Builds a
weighted combination of untreated units to construct the counterfactual a
single treated unit would have followed.

**Phase 6 — Regression Discontinuity.** A simulated scholarship threshold.
Surfaces a false-positive McCrary density test (p = 0.000) from a simplified
bin-based implementation — documented as a lesson in how easy it is to
misread test output.

**Phase 7 — Double Machine Learning.** Same LaLonde + CPS-3 data as Phases 3
and 4. Replaces AIPW's linear nuisance models with cross-fitted gradient
boosting and a Neyman-orthogonal score. ATT = $1,499 (83.6% of benchmark,
best point estimate in the series), CI [−$492, $3,519] — crossing zero. Also
documents a precision trap: the Full CPS path gives a *significant* p = 0.020,
which is false precision from large N masking an effective sample size of
2.3%.

**Phase 8 — DoWhy + EconML (Capstone).** The 401(k) eligibility dataset (n =
9,915, Chernozhukov et al. 2018), with genuine overlap (AUC = 0.699). Both ATE
and ATT estimated by seven methods. Six converge within an $874 band, ~84% of
the $9,000 benchmark. DoWhy's backdoor regression passes all three refutation
tests (placebo, random common cause, data subset) — its identification is
correct — yet underestimates by 44% because a single linear coefficient can't
represent the heterogeneous effects CausalForest reveals (CATE ranges from
$642 to $20,831 across the income distribution).

## Cross-series estimates at a glance

All LaLonde-based phases (3, 4, 7) target ATT, comparable to the $1,794 RCT
benchmark:

| Method | ATT | 95% CI | Recovery |
|--------|-----|--------|----------|
| PSM (Phase 3) | $300 | [−$1,819, $2,064] | 17% |
| IPW trimmed (Phase 4) | $1,212 | [−$141, $2,718] | 68% |
| AIPW (Phase 4) | $476 | [−$1,132, $2,084] | 27% |
| DML (Phase 7) | $1,499 | [−$492, $3,519] | 84% |

Phase 8 (401k, benchmark ATE ≈ $9,000):

| Method | ATE | ATT |
|--------|-----|-----|
| DoWhy backdoor | $5,028 | $5,028 |
| S-Learner | $7,495 | $8,183 |
| T-Learner | $8,011 | $9,132 |
| X-Learner | $8,015 | $9,552 |
| NonParamDML | $8,052 | $10,312 |
| LinearDML | $8,185 | $10,716 |
| CausalForest | $8,369 | $11,238 |
| **Mean** | **$7,594** | **$9,166** |

## Repository structure

```
applied-causal-inference/
├── notebooks/
│   ├── ABTest.py
│   ├── DiD.py
│   ├── PSM.py
│   ├── IPW_AIPW.py
│   ├── SyntheticControl.py
│   ├── RDD.py
│   ├── DoubleML.py
│   └── Phase8_DoWhy_EconML.py
├── assets/
│   └── plots/
│       ├── phase1/ ... phase8/
├── index.html
└── README.md
```

## Requirements

- Databricks Runtime 13.0+
- Python 3.9+
- `pip install scipy statsmodels scikit-learn mlflow dowhy econml doubleml`

## Series themes

Three ideas recur across all eight phases:

1. **Identification precedes estimation.** No estimator — however
   sophisticated — fixes a population that doesn't overlap. Phase 7's Full
   CPS result (statistically significant, and wrong) is the clearest example.

2. **The estimand is a choice, not a default.** ATE and ATT are different
   questions with different answers whenever treatment effects are
   heterogeneous. Phase 8 is the first to compute both side by side and show
   the gap directly.

3. **Convergence across methods with different assumptions is stronger
   evidence than any single confidence interval — and divergence points to
   which assumption is being violated, not which number is correct.**
