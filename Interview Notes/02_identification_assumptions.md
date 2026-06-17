# 02 — Identification Assumptions

Every causal estimate rests on assumptions that cannot be fully verified from data alone. This file catalogues the major ones: what each assumption claims, why the method needs it, how it typically fails, and what (if anything) can be checked empirically.

---

### Unconfoundedness / Ignorability / Exchangeability

**What it means:** Given a set of observed covariates X, treatment assignment is independent of the potential outcomes: {Y(0), Y(1)} ⊥ D | X. Equivalently — among units with the same X, treatment was "as good as randomly assigned." All three names (unconfoundedness, ignorability, conditional exchangeability) refer to the same condition; "selection on observables" is another common phrase for it.

**Why it's needed:** This is the core assumption behind essentially every observational method that doesn't rely on an instrument or a discontinuity — PSM, IPW, AIPW, DML, regression adjustment, DoWhy's backdoor estimator. Without it, E[Y | D=1, X] − E[Y | D=0, X] doesn't equal the causal effect — it's contaminated by whatever unobserved factors drove both treatment and outcome.

**How it fails in practice:** Whenever there's an *unobserved* confounder — a variable affecting both treatment and outcome that isn't in X. Classic example: ability or motivation affecting both whether someone enrolls in a training program and their later earnings, where "ability" is never measured.

**How it's checked:** It *cannot* be directly tested — by definition, you'd need to observe the unobservable. What can be done: refutation tests (add a random/placebo confounder and check the estimate doesn't move — Phase 8's DoWhy refutation suite), sensitivity analysis (how strong would an unobserved confounder need to be to overturn the result — e.g., Rosenbaum bounds, E-values), and theoretical/domain argument for why no major confounder is likely missing.

---

### Positivity / Overlap / Common Support

**What it means:** Every unit has a non-zero probability of receiving *either* treatment level, given its covariates: 0 < P(D=1 | X=x) < 1 for all x in the population. Equivalently — for every covariate profile, both treated and control units actually exist in the data.

**Why it's needed:** Methods that compare treated and control units (matching, weighting, residualisation) need something to compare *to*. If some covariate region has only treated units (or only controls), there's no empirical counterfactual there — the method either drops those units (changing the estimand) or extrapolates (introducing model-dependence).

**How it fails in practice:** When treatment assignment is strongly determined by covariates — e.g., a program targeted specifically at high-need individuals, so "high need + untreated" essentially doesn't exist in the data. Manifests as a propensity score model achieving very high AUC (near-perfect separation between treated and control).

**How it's checked:** Propensity score AUC (rule of thumb: >0.90 borderline, >0.95 likely failure), histogram overlap of propensity scores by group, effective sample size (ESS) after reweighting — a low ESS relative to N signals that only a small, possibly unrepresentative, subset of the data is doing the identification work.

**Important nuance:** Poor overlap doesn't just widen confidence intervals — it can produce *misleadingly tight* ones. When N is very large but overlap is poor, standard-error formulas that scale with 1/√N can shrink even as the effective sample size collapses, producing a confident-looking estimate built on almost no real comparison data.

---

### Consistency (SUTVA's "well-defined treatment" component)

**What it means:** The potential outcome Y(d) that would be observed under treatment level d is the same regardless of *how* d was assigned or delivered — i.e., the observed outcome under the treatment actually received equals the potential outcome for that treatment level: Y_obs = Y(D).

**Why it's needed:** Without it, "the effect of treatment" is not a well-defined quantity — there could be many different "treatments" lumped under one label, each with a different effect, and the estimand would be some uninterpretable blend of them.

**How it fails in practice:** A "job training program" that's delivered with very different content/intensity/quality at different sites is really multiple different treatments coded as one variable. Estimating "the effect of training" then estimates a blend that doesn't describe any actual program.

---

### Parallel Trends (Difference-in-Differences)

**What it means:** In the absence of treatment, the treated and control groups would have followed the *same trend* over time — the gap between them would have stayed constant. DiD attributes any *change* in the gap, after treatment begins, to the treatment itself.

**Why it's needed:** DiD's entire logic is "compare the change over time in the treated group to the change over time in the control group" — this only isolates the treatment effect if the two groups' trajectories would otherwise have moved in parallel.

**How it fails in practice:** When the treated group was already on a different trajectory before treatment — e.g., a region selected for a policy *because* it was already declining faster than other regions. Then post-treatment divergence reflects the pre-existing trend, not the policy.

**How it's checked:** Pre-trends test — plot (or formally test) the gap between groups in the pre-treatment periods. If the gap is stable before treatment and only changes after, parallel trends is plausible. A statistically significant pre-trend is evidence against the assumption (though absence of a significant pre-trend is not proof it holds — pre-trend tests are often underpowered).

---

### Exclusion Restriction (Instrumental Variables)

**What it means:** The instrument Z affects the outcome Y *only* through its effect on treatment D — there is no direct path Z → Y that bypasses D, and no path through any unobserved confounder of Z and Y.

**Why it's needed:** IV's logic is to use variation in Z as a proxy for "as-if-random" variation in D. If Z also affects Y directly, then the Z–Y association reflects both the indirect effect (through D, the thing we want) and the direct effect (a contaminant) — and these can't be separated without further assumptions.

**How it fails in practice:** A classic failure: using "distance to the nearest college" as an instrument for "years of education" to estimate the effect on earnings — if living near a college also correlates with local labor market conditions that independently affect earnings, the exclusion restriction is violated.

**How it's checked:** Cannot be tested directly (same fundamental issue as unconfoundedness) — requires a substantive, defensible argument for *why* no direct path exists. Overidentification tests (with multiple instruments) can detect *some* violations but not all.

---

### Monotonicity (Instrumental Variables)

**What it means:** The instrument moves everyone's treatment status in the *same direction* (or not at all) — there are no "defiers," units who would do the opposite of what the instrument nudges them toward. Formally: D(Z=1) ≥ D(Z=0) for every unit (or ≤, consistently).

**Why it's needed:** Without monotonicity, the IV estimate becomes a weighted average of effects for compliers (who move with the instrument) and defiers (who move against it), with weights that can produce a LATE that doesn't represent any coherent subpopulation — and can even have the wrong sign.

**How it fails in practice:** An instrument that has heterogeneous effects across the population — e.g., a policy nudge that encourages some people toward treatment but, due to a behavioral quirk (reactance), pushes a small subgroup away from it.

---

### No Anticipation (Event Studies / DiD with known treatment timing)

**What it means:** Units do not change their behavior *in anticipation* of a future treatment — outcomes in period t are unaffected by treatment that hasn't happened yet but is known to be coming.

**Why it's needed:** If units anticipate treatment and adjust behavior beforehand, the "pre-treatment" period is contaminated — it no longer represents the true counterfactual baseline, biasing the estimated effect (often toward zero, since some of the "effect" already happened pre-treatment).

**How it fails in practice:** A tax change announced months before implementation — firms and individuals adjust behavior immediately upon announcement, not at the implementation date. Using the implementation date as "treatment start" misses the anticipatory response.

---

### No Interference / No Spillovers (the other half of SUTVA)

**What it means:** Already introduced under SUTVA — restated here because it's an *identification* assumption specifically for estimating ATE/ATT from data where units might interact. One unit's potential outcomes don't depend on other units' treatment status.

**How it fails in practice:** Two-sided markets, social networks, and general-equilibrium settings are the classic violation zones. A jobs program that helps some workers find jobs may make it *harder* for untreated workers to find jobs (displacement) — the "control group" outcome is itself affected by how many other units were treated, so it no longer represents the true counterfactual.

**How it's checked:** Often addressed by design (cluster-randomization to keep interacting units in the same arm) rather than by a statistical test. Detecting interference after the fact typically requires explicit network/spatial structure in the data.
