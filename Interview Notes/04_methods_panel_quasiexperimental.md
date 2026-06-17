# 04 — Methods: Panel and Quasi-Experimental Designs

---

### Difference-in-Differences (DiD)

**What it means:**
- Compares the *change over time* in outcomes between a treated group and a control group.
- Requires at least two periods (pre and post) and two groups (treated and untreated).
- Estimator: (Y_treated,post − Y_treated,pre) − (Y_control,post − Y_control,pre)
- The "difference of differences" cancels out any time-invariant differences between groups and any common time trend shared by both groups.
- With panel data and many units/periods, implemented as a two-way fixed effects (TWFE) regression: unit fixed effects + time fixed effects + treatment indicator.

**Assumptions:**
- Parallel trends — absent treatment, the gap between groups would have stayed constant over time.
- No anticipation — units don't change behavior before treatment actually starts.
- SUTVA — no spillovers between treated and control groups.
- (For TWFE with staggered adoption) Treatment effects are homogeneous across units and over time — this assumption is now known to be commonly violated and can produce badly biased estimates ("negative weighting" problem in modern DiD literature).

**When to use:**
- Panel or repeated cross-sectional data with a clear pre/post structure and a plausible untreated comparison group.
- When you can plot pre-treatment trends for both groups and they look parallel.
- When treatment timing is the same for all treated units (the "vanilla" 2x2 DiD case is the safest).

**When not to use:**
- Single cross-section with no time dimension — there's no "before" to compare.
- Treated and control groups were already diverging before treatment (pre-trends test fails).
- Staggered treatment adoption across units/times without using one of the modern heterogeneity-robust estimators (Callaway-Sant'Anna, Sun-Abraham, etc.) — plain TWFE can be severely biased here.
- Spillovers are likely (the "control" group is actually affected by the treatment given to others).

---

### Synthetic Control

**What it means:**
- Constructs a "synthetic" counterfactual for a single treated unit by taking a weighted combination of untreated "donor" units.
- Weights are chosen so the synthetic control's *pre-treatment* outcome (and covariate) trajectory closely matches the treated unit's actual pre-treatment trajectory.
- The post-treatment gap between the treated unit and its synthetic counterpart is the estimated treatment effect.
- Originally developed for comparative case studies (one treated state/country vs. a panel of untreated ones).

**Assumptions:**
- The donor pool contains units that, in combination, can approximate the treated unit's counterfactual trajectory (no donor unit was itself affected by the treatment, or by something correlated with it — no spillover into the donor pool).
- The pre-treatment fit is good — if the synthetic control can't match the pre-treatment trajectory well, post-treatment gaps are not interpretable as treatment effects.
- No anticipation effects contaminating the pre-treatment period used for fitting weights.

**When to use:**
- A single (or very small number of) treated unit(s), with a panel of plausible untreated comparison units over time.
- Case-study-style policy evaluation — "what would have happened to this state/country/firm without the policy?"
- When a good pre-treatment fit is achievable — many periods of pre-treatment data and a donor pool that spans the relevant covariate space.

**When not to use:**
- Individual-level cross-sectional data — there's no "donor panel" of comparable individual units to construct weights from.
- Many treated units (synthetic control doesn't scale gracefully to a "treated group" the way DiD does — though extensions exist).
- Pre-treatment fit is poor regardless of weighting — the method has nothing to anchor the counterfactual to.
- Donor units are themselves affected by the treatment or a common shock around the treatment date (contaminates the counterfactual).

---

### Regression Discontinuity Design (RDD)

**What it means:**
- Exploits a situation where treatment is assigned based on whether a continuous "running variable" crosses a known threshold (e.g., test score ≥ 70 → scholarship).
- Units just above and just below the threshold are assumed to be otherwise similar — the discontinuity in treatment at the cutoff, combined with a (assumed) smooth/continuous relationship between the running variable and the outcome absent treatment, lets the jump in outcome at the cutoff be attributed to treatment.
- **Sharp RDD** — treatment is deterministically assigned by the cutoff (everyone above gets treated, everyone below doesn't).
- **Fuzzy RDD** — crossing the cutoff changes the *probability* of treatment but doesn't fully determine it (treatment uptake is imperfect) — estimated via a local IV/2SLS using the threshold-crossing indicator as the instrument.
- The estimand is a LATE — the effect *at the threshold*, for units near the cutoff. It does not generalize to units far from the cutoff.

**Assumptions:**
- Continuity — in the absence of treatment, the relationship between the running variable and the outcome would be smooth/continuous through the cutoff (no other "jump" happens to coincide with the treatment cutoff).
- No manipulation of the running variable — units cannot precisely control which side of the cutoff they land on (e.g., a student can't perfectly target a test score of exactly 70).
- (Fuzzy RDD additionally) The same exclusion restriction and monotonicity assumptions as IV, applied locally at the threshold.

**When to use:**
- Treatment assignment follows a known, verifiable rule based on a continuous variable crossing a threshold.
- The running variable cannot be precisely manipulated by units (or by administrators) around the cutoff.
- You're specifically interested in the effect *near the threshold* — e.g., for a policy decision about where to set a similar cutoff.

**When not to use:**
- Treatment assignment is a discrete decision with no underlying continuous running variable (e.g., a binary employer decision like offering a benefit plan) — there's no "threshold" to exploit.
- The running variable can be manipulated — e.g., self-reported income near an eligibility cutoff, where people may misreport to qualify (test with a density/manipulation check such as the McCrary test — but be cautious: simplified bin-based implementations of this test can produce false positives even with no real manipulation, simply from normal curvature in the density near the cutoff).
- You need an estimate that generalizes to the full population, not just units near the cutoff — RDD's LATE has narrow external validity by construction.
- Insufficient data density near the cutoff — local estimation requires enough observations close to the threshold for the bandwidth/polynomial fit to be reliable.

*There is also Triple Difference (DDD).*

---

### Event Study Design

**What it means:**
- A generalization of DiD that estimates treatment effects separately for each period relative to treatment timing (t = -3, -2, -1, 0, +1, +2, +3, ...) rather than collapsing to a single pre/post average.
- Produces a plot of estimated effects over event-time — pre-treatment coefficients near zero support parallel trends; post-treatment coefficients show the dynamic path of the effect (does it appear immediately, grow, fade?).

**Assumptions:**
- Same as DiD (parallel trends, no anticipation, SUTVA) — event studies are a diagnostic *and* estimation tool for these assumptions, not a way around them.

**When to use:**
- You want to visually/formally assess the pre-trends assumption underlying a DiD design.
- The treatment effect is plausibly dynamic — building up over time, or fading — and a single pre/post average would obscure this pattern.

**When not to use:**
- Very few pre-treatment periods — not enough data to assess pre-trends meaningfully.
- Staggered treatment timing across units without using a heterogeneity-robust event-study estimator — naive event-study regressions with unit and time fixed effects can suffer the same "negative weighting" bias as TWFE DiD under staggered adoption.

---

### Instrumental Variables (IV) / Two-Stage Least Squares (2SLS)

**What it means:**
- Uses a variable Z (the "instrument") that affects treatment D but has no direct effect on outcome Y except through D, to isolate "as-if-random" variation in D.
- **2SLS mechanics:** First stage — regress D on Z (and covariates) to get predicted treatment D̂. Second stage — regress Y on D̂ (and covariates). The coefficient on D̂ is the IV estimate.
- **Wald estimator** (single binary instrument, no covariates): θ_IV = (E[Y|Z=1] − E[Y|Z=0]) / (E[D|Z=1] − E[D|Z=0]) — the "reduced form" effect of the instrument on the outcome, scaled by the instrument's effect on treatment uptake.
- The result is a LATE — the effect for "compliers" (units whose treatment status responds to the instrument).

**Assumptions:**
- Relevance — the instrument actually affects treatment (E[D|Z=1] ≠ E[D|Z=0]); a "weak instrument" (small first-stage effect) inflates variance and can bias 2SLS toward the (potentially confounded) OLS estimate.
- Exclusion restriction — instrument affects Y only through D (see file 02).
- Monotonicity — no defiers (see file 02).
- Independence — the instrument itself is "as good as randomly assigned" with respect to potential outcomes (often the hardest assumption to defend).

**When to use:**
- A plausible source of "natural experiment" variation exists — a policy rule, lottery, geographic discontinuity, or similar mechanism that shifts treatment for reasons unrelated to the outcome.
- Unconfoundedness (selection on observables) is not credible — there's a known or suspected *unobserved* confounder, and IV is one of the few tools that can address this.

**When not to use:**
- No credible instrument exists — a weak or implausible instrument is often worse than an observational method with a defensible unconfoundedness argument, because IV with a bad instrument can be badly biased *and* falsely appear precise.
- The exclusion restriction is hard to defend — instruments that plausibly affect the outcome through channels other than the treatment of interest invalidate the entire approach, and this cannot be tested directly.
- The policy-relevant estimand is ATE or ATT for the full population, and the complier subpopulation (LATE) is known or suspected to be very different from that population — e.g., an instrument where compliers are a small, unusual slice of the data.
