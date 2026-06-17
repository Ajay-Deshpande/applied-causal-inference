# 06 — The DoWhy Identification Framework

DoWhy (and similar frameworks) structure causal analysis into four explicit steps: **Model → Identify → Estimate → Refute**. The framework's contribution isn't a new estimator — it's making the assumptions behind *any* estimator explicit and testable.

---

### Step 1 — Model (Declare the DAG)

**What it means:**
- Encode the causal assumptions as a directed acyclic graph: which variables affect which others, including treatment, outcome, and all covariates believed to be confounders, mediators, or instruments.
- This DAG *is* the assumption set — everything downstream depends on it being a reasonable representation of the data-generating process.

**When to use:**
- Always, as a first step — even an informal sketch of the DAG clarifies which variables play which roles (confounder vs. mediator vs. collider) before any adjustment decisions are made.

**When not to use / limitations:**
- The DAG cannot verify itself — a missing edge (an unmeasured confounder you didn't draw) or a wrong-direction edge produces a DAG that looks complete but isn't. Domain expertise is doing the real work here; the DAG just makes it explicit and auditable.

---

### Step 2 — Identify (Derive the Estimand)

**What it means:**
- Given the DAG, determine *whether* the causal effect is identifiable from observational data, and if so, *which* adjustment formula recovers it.
- Most commonly resolves to the **backdoor criterion** (see file 02): find a set Z of variables such that conditioning on Z blocks all backdoor paths from treatment to outcome.

**Formula (backdoor adjustment):**

  `E[Y|do(D=d)] = Σ_z E[Y|D=d, Z=z]·P(Z=z)`

- **do(D=d)**: Pearl's "do-operator" — denotes the *interventional* distribution (what would happen if D were *set* to d by intervention), as opposed to the *observational* conditional distribution P(Y|D=d) (what we observe when D *happens* to equal d, which may be confounded).
- **Z**: the adjustment set identified by the backdoor criterion — typically the observed confounders.
- The formula says: the interventional outcome distribution equals the observational outcome distribution *conditional on D and Z*, averaged over the observed distribution of Z. This is the formal justification for "control for confounders and average."

**When to use:**
- After declaring the DAG — this step tells you *whether your causal question can be answered at all* given the graph, before you commit to an estimation method.

**When not to use / limitations:**
- If the identification step reports the effect is *not identifiable* (e.g., an unobserved confounder with no valid adjustment set and no instrument), no amount of clever estimation in step 3 will fix this — the problem is in the DAG/data, not the estimator.

---

### Step 3 — Estimate

**What it means:**
- Apply a concrete estimator (linear regression, propensity matching, IPW, DML, etc.) to compute the identified estimand from Step 2.
- This is where any of the methods in files 03–05 plug in — DoWhy is agnostic to *which* estimator you use, as long as it targets the identified estimand.

**When to use:**
- Once identification succeeds — choose the estimator based on data characteristics (overlap, sample size, outcome distribution) as covered in files 03–05.

**When not to use / limitations:**
- A common pitfall: using a simple/linear estimator here (e.g., linear regression for backdoor adjustment) when the *identification* is correct but the *functional form* assumed by the estimator is wrong (e.g., linear regression on a relationship with strong nonlinear confounding). The estimate can be biased even though the adjustment set is exactly right — identification and estimation are separate questions, and refutation tests (Step 4) mostly probe the former, not the latter.

---

### Step 4 — Refute

**What it means:**
- A suite of tests that probe the *robustness* of the Step 3 estimate to violations of the Step 1/2 assumptions — none of these tests can *prove* the assumptions hold, but each can provide evidence *against* them (or fail to find evidence against them).

---

#### Placebo Treatment Refuter

**What it means:**
- Replace the real treatment variable with a randomly permuted (placebo) version, and re-run the estimation.
- A placebo treatment has, by construction, no real causal relationship with the outcome — so a well-specified estimator should produce an effect estimate close to zero.

**Formula / procedure:**
- Permute D across units (breaking its real relationship with X and Y) → D_placebo
- Re-estimate: `θ̂_placebo = Estimator(Y, D_placebo, X)`
- Compare `θ̂_placebo` to the original `θ̂` — ideally `|θ̂_placebo| << |θ̂|`.

**When to use:**
- As a general sanity check on any estimation pipeline — if the placebo estimate is *not* close to zero, the estimator is picking up something other than a causal effect (e.g., overfitting, leakage, or a flawed adjustment set producing spurious associations regardless of the treatment variable's actual values).

**When not to use / limitations:**
- Passing this test is necessary but not sufficient — it rules out gross overfitting/leakage but says nothing about whether the *real* treatment effect estimate is biased by an unobserved confounder that happens to be correlated specifically with the real (not placebo) treatment assignment.

---

#### Random Common Cause Refuter

**What it means:**
- Add a randomly generated variable (independent of everything) to the model as if it were an additional confounder, and re-estimate.
- Since this variable is pure noise, it shouldn't change the estimate if the estimator is stable.

**When to use:**
- To check sensitivity to "irrelevant" additional covariates — an estimator whose result shifts substantially when an unrelated random variable is added is a sign of instability (e.g., overfitting in a high-dimensional nuisance model, or a matching/weighting procedure sensitive to the exact covariate set).

**When not to use / limitations:**
- This test uses a variable known to be *unrelated* to treatment or outcome — it doesn't address the central concern (an unobserved confounder that *is* related to both). It's a stability check, not an unconfoundedness check.

---

#### Data Subset Refuter

**What it means:**
- Re-estimate the effect on a random subset (e.g., 80%) of the data, repeated multiple times, and compare to the full-sample estimate.

**When to use:**
- To assess whether the estimate is being driven by a small number of influential observations, or is stable across resamples of the data — a form of informal stability/robustness check related to (but simpler than) a full bootstrap.

**When not to use / limitations:**
- High variance across subsets indicates instability but doesn't diagnose *why* — could be small sample size generally, could be a few high-leverage points, could be poor overlap concentrating identification in a small slice of the data (see the "precision trap" discussion in file 05/08).

---

#### Sensitivity Analysis: Rosenbaum Bounds and E-values

**What it means:**
- Rather than testing with a *specific* placebo/random variable, sensitivity analysis asks: *how strong would an unobserved confounder need to be* — in terms of its association with both treatment and outcome — *to overturn the conclusion* (e.g., reduce the effect to zero, or flip its sign)?

**Rosenbaum bounds:**
- Originally developed for matched-pairs designs. Introduces a sensitivity parameter Γ representing how much more likely a unit could be to receive treatment due to an unobserved confounder (an odds-ratio-scale parameter). Reports the range of Γ over which the conclusion (e.g., statistical significance) still holds.
- **Γ**: if Γ=2, it means even if an unobserved factor made one unit twice as likely as another (matched) unit to receive treatment, the conclusion would still hold — small Γ thresholds mean conclusions are fragile to even modest unobserved confounding.

**E-value:**
- The minimum strength of association (on the risk-ratio scale), that an unmeasured confounder would need to have with *both* treatment and outcome, *above and beyond* the measured covariates, to fully explain away the observed effect.
- A large E-value means a very strong (and thus less plausible) unobserved confounder would be required to nullify the result — supporting (but not proving) robustness. A small E-value means even a weak unobserved confounder could overturn the conclusion.

**When to use:**
- Whenever unconfoundedness is the load-bearing assumption (which is most observational analyses) — sensitivity analysis converts "we assume no unobserved confounders" into "here's how strong an unobserved confounder would need to be to matter, and you can judge for yourself how plausible that is."
- Particularly valuable for communicating to stakeholders/reviewers — it reframes an unfalsifiable assumption into a quantitative, debatable claim.

**When not to use / limitations:**
- Sensitivity analysis tells you *how much* unobserved confounding would be needed to overturn results — it cannot tell you whether such confounding actually *exists*. A result can have a reassuringly large E-value and still be wrong if the real unobserved confounder happens to be unusually strong.
- Different sensitivity frameworks (Rosenbaum bounds, E-values, and others) make different assumptions about the *form* of the unobserved confounding and aren't always directly comparable to each other.
