# Applied Causal Inference

A practitioner's series implementing causal inference methods from scratch —
on real datasets, in Databricks, with MLflow tracking.

**Read the series:** https://Ajay-Deshpande.github.io/applied-causal-inference/

**Medium:** [Article Link](https://medium.com/@ajay-deshpande/why-most-data-science-doesnt-answer-the-question-you-re-asking-508aad563418)

## Notebooks

| Phase | Method | Dataset | Status |
|-------|--------|---------|--------|
| 1 | A/B Testing / RCT | Simulated e-commerce | ✓ Live |
| 2 | Difference-in-Differences | LaLonde (1986) | Coming |
| 3 | Propensity Score Matching | LaLonde (1986) | Coming |
| 4 | Synthetic Control | Simulated state panel | Coming |
| 5 | Regression Discontinuity | Simulated scholarship | Coming |
| 6 | DoWhy + Causal Graphs | LaLonde (1986) | Coming |
| 7 | Final Comparison | LaLonde (1986) | Coming |

## Requirements
- Databricks Runtime 13.0+
- Python 3.9+
- `pip install scipy statsmodels mlflow dowhy econml`
