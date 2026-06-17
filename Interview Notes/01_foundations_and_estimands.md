# 01 — Foundations and Estimand Types

---

## Part A: Foundational Concepts

### Potential Outcomes Framework (Rubin Causal Model)

**What it means:** Every unit i has two potential outcomes — Y_i(1) if treated, Y_i(0) if not treated. The causal effect for that unit is Y_i(1) − Y_i(0). We only ever observe one of these for any given unit:

Y_obs = D · Y(1) + (1−D) · Y(0)

This is the "fundamental problem of causal inference" — the other potential outcome is a counterfactual, never observed. All causal inference methods are strategies for estimating what that missing outcome would have been, on average, across some group of units.

---

### SUTVA (Stable Unit Treatment Value Assumption)

**What it means:** Two conditions bundled together:
1. **No interference** — one unit's outcome doesn't depend on another unit's treatment assignment (no spillovers).
2. **No hidden variation in treatment** — "treated" means the same thing for every unit; there isn't a different "version" of treatment for different people.

**Why it matters:** Violated constantly in practice. Vaccination has herd-immunity spillovers (interference). A "training program" delivered differently at two sites is hidden variation. Most causal methods implicitly assume SUTVA holds; if it doesn't, the estimand itself may be ill-defined.

---

### Confounder

**What it means:** A variable that causally affects *both* treatment assignment and the outcome. Creates a spurious treatment-outcome association if not adjusted for. Graphically: T ← X → Y (a "fork").

**Example:** Income affects both whether someone has access to a retirement plan and how much they save — income is a confounder of plan-access → savings.

---

### Mediator

**What it means:** A variable on the causal pathway between treatment and outcome: T → M → Y. Conditioning on a mediator blocks part of the very effect you're trying to measure — it "explains away" the mechanism rather than removing bias.

**Why it matters:** Including a mediator as a "control variable" is a common mistake — it doesn't reduce confounding, it reduces the measured effect, because you're holding constant something the treatment itself changes.

**Example:** Treatment is attending a training program. If we want measure causal pathway between treatment (attended a training program) and income we shouldn't add skills gained through training program. Here skills will be a mediator which will block or reduce the effect of treatment variable

---

### Collider

**What it means:** A variable causally affected by *two* other variables of interest — two arrows point *into* it: T → C ← Y (or X → C ← Y for any two variables X, Y). Conditioning on (or selecting a sample based on) a collider *induces* a spurious association between its causes, even if none exists causally.

**Example (Berkson's paradox):** Among hospitalized patients, two unrelated diseases can appear negatively correlated, because having either one increases hospitalization (the collider) — conditioning on "hospitalized" creates the spurious link.

**Why it matters:** This is the opposite failure mode from a confounder — with a confounder you *must* condition to remove bias; with a collider, conditioning *creates* bias.

---

### Moderator / Effect Modifier

**What it means:** A variable that doesn't cause treatment or outcome directly, but changes the *size or direction* of the treatment effect — it's a source of treatment effect heterogeneity.

**Example:** Income as an effect modifier — the effect of 401(k) eligibility on savings is much larger for higher earners (they have more to redirect into savings) even if income doesn't directly *cause* eligibility status to change after the fact.

**Note:** A variable can be both a confounder *and* an effect modifier simultaneously — these are not mutually exclusive roles.

---

### Causal DAG (Directed Acyclic Graph)

**What it means:** A graphical encoding of causal assumptions — nodes are variables, directed edges represent direct causal effects, and the graph has no cycles (nothing causes itself through a chain). DAGs make assumptions explicit and let you reason mechanically about which variables need to be controlled for.

**Advantages:** Forces explicit statement of assumptions; enables automated identification (backdoor/frontdoor criteria, do-calculus); supports refutation/sensitivity testing.

**Disadvantages:** The graph itself is an assumption — if it's wrong (missing an edge, wrong direction), the "correct" adjustment set derived from it will also be wrong. A DAG cannot verify itself.

---

### Backdoor Path / Backdoor Criterion

**What it means:** A backdoor path is any non-causal path between treatment T and outcome Y that starts with an arrow *into* T (i.e., flows through a confounder). The **backdoor criterion** says a set of variables Z is sufficient for adjustment if: (1) no variable in Z is a descendant of T, and (2) Z blocks every backdoor path between T and Y. If satisfied, the causal effect is identified as E[Y | T, Z] averaged appropriately over Z.

**Why it matters:** This is the formal version of "control for confounders" — most regression-adjustment, matching, and weighting methods are implementations of backdoor adjustment.

```
      C (Confounder)
     ↙             ↘
T (Treatment) → Y (Outcome)

Backdoor path:
T ← C → Y

Causal path:
T → Y
```
---

### Frontdoor Criterion

**What it means:** An identification strategy for when treatment and outcome share an *unobserved* confounder, but there exists a mediator M that (a) fully carries the effect of T on Y, and (b) is itself unconfounded with Y given T. The effect can then be identified via the T → M → Y path even though the direct confounding of T and Y is unobserved.

**Why it matters:** Rare in practice — requires a mediator that fully mediates the effect and is itself clean. Mostly useful as a conceptual tool for understanding what "identification" can mean beyond simple adjustment.

**Example:**
- T: Ad Exposure
- M: Product Awareness
- Y: Purchase
- U: Customer Interest (unobserved)

```
U → T
U → Y

T → M
M → Y
```
*We cannot adjust for U because it's unobserved.*

But suppose:
- All effect of T on Y goes through M.
- No confounder exists between T and M.
- After controlling for T, there is no confounder between M and Y.

Then we can identify:
```T → M → Y```

even though:
```T ← U → Y```

---

### do-calculus

**What it means:** Pearl's formal system (three inference rules) for determining whether — and how — a causal query P(Y | do(X)) can be rewritten in terms of purely observational quantities P(Y | X, ...), given a DAG. The backdoor and frontdoor criteria are special cases of do-calculus rules.

**Why it matters:** It's the theoretical foundation underlying tools like DoWhy — when DoWhy says an estimand is "identified," it's running (a restricted form of) do-calculus against your declared graph.

---

### Selection Bias vs. Confounding Bias

**What it means:** Two distinct sources of spurious association:
- **Confounding bias** arises from a common cause of treatment and outcome (a "fork": T ← X → Y). Fixed by adjusting for X (the common cause).
- **Selection bias** arises from conditioning on (or non-randomly sampling based on) a collider, or any variable downstream of both T and Y. Fixed by *not* conditioning on it — adjustment makes it worse, not better.

**Why it matters:** The fix for one is the opposite of the fix for the other. Misdiagnosing which type of bias you have leads to "corrections" that introduce new bias.

---
---

## Part B: Estimand Types

### ATE — Average Treatment Effect

**What it means:** E[Y(1) − Y(0)], averaged over the *entire population* of interest. "What would happen, on average, if everyone in this population were treated vs. if no one were?"

**Advantages:** Answers the question relevant to universal policy decisions — should this intervention be rolled out to everyone?

**Disadvantages:** Can be diluted by heterogeneity — a large positive effect for one subgroup and a near-zero effect for another can average to a modest ATE that doesn't describe either subgroup well. Also may not be the policy-relevant question if the intervention will only ever reach a subset of the population.

---

### ATT — Average Treatment Effect on the Treated

**What it means:** E[Y(1) − Y(0) | D=1] — the average effect *among those who actually received treatment*. "What was the effect, for the people who got it?"

**Advantages:** Directly answers "did this program work for its participants?" — the natural question when evaluating an existing, non-randomly-assigned program.

**Disadvantages:** Not generalizable to non-participants. ATT ≠ ATE whenever treatment assignment is correlated with the potential outcomes or with effect-modifying covariates (e.g., people who self-select into a program may benefit more or less than average).

*Other version: ATC — Average Treatment Effect on the Controls*

---

### CATE — Conditional Average Treatment Effect

**What it means:** E[Y(1) − Y(0) | X=x] — the treatment effect as a *function of covariates*, capturing effect heterogeneity. ATE = E_X[CATE(X)] and ATT = E[CATE(X) | D=1] — ATE and ATT are both weighted averages of the same underlying CATE function, just with different weighting distributions over X.

**Advantages:** Reveals *who* benefits most — enables targeting/personalization. Often the most policy-actionable estimand: "the effect is large for high-income workers and near zero for low-income workers" is more useful than a single pooled number.

**Disadvantages:** Requires substantially more data for reliable estimation than a single pooled effect — you're estimating a function, not a number. Inference is harder: analytic confidence intervals for CATE-based estimators (e.g., causal forests) can be very wide or unstable even when the point estimates are reasonable, often requiring bootstrap as a cross-check.

---

### ITT — Intention to Treat

**What it means:** The effect of being *assigned* to treatment, regardless of whether the unit actually complied with / took up the treatment. Standard in RCTs with imperfect compliance.

**Advantages:** Preserves the unbiasedness of randomization — assignment is still random even if uptake isn't, so ITT is a clean causal estimate of *the policy of offering/assigning* treatment.

**Disadvantages:** Dilutes the effect of *actually receiving* treatment. If only 60% of the assigned group complies, ITT understates the per-protocol effect on compliers — it answers "what if we offered this to everyone" but not "what is the effect of actually getting it."

---

### LATE — Local Average Treatment Effect

**What it means:** In an instrumental variables (IV) setting, the average treatment effect *for compliers* — units whose treatment status would change if the instrument's value changed. E[Y(1) − Y(0) | complier].

**Advantages:** Identifiable via IV even in the presence of unobserved confounding between treatment and outcome, under the exclusion restriction (the instrument affects Y only through T) and monotonicity (the instrument doesn't push anyone's treatment status in opposite directions — no "defiers").

**Disadvantages:** Only describes the subpopulation of compliers — an unobservable group that may be a small and unrepresentative slice of the population. "Local" external validity: the effect for compliers may not generalize to always-takers, never-takers, or the population as a whole. Also the estimand most commonly produced by sharp/fuzzy regression discontinuity designs — there, "compliers" are units near the threshold, so RDD's LATE is local in *covariate space* as well as in compliance type.

---

### Marginal vs. Conditional Effects

**What it means:** A *marginal* effect is averaged over the population (ATE-like) — it answers "on average, across everyone." A *conditional* effect is evaluated at specific covariate values (CATE-like) — "for someone with these characteristics." In nonlinear models (logistic regression, GBM, etc.), these can differ substantially: the "average of individual effects" (marginal effect at each observation, then averaged) is not the same as the "effect evaluated at the average covariate values" (effect at the mean) — the two coincide only for linear models.

**Why it matters:** A frequent interview trap — "the marginal effect" is ambiguous language unless you specify which of these two quantities is meant, and the gap between them grows with the degree of nonlinearity in the underlying model.
