# 05 — Methods: Machine-Learning-Based Estimators

---

### Double Machine Learning (DML) — Robinson Partially Linear Regression (PLR)

**What it means:**
- Models the outcome and treatment as two equations sharing the same covariates:

  `Y = θ·D + g(X) + ε`
  `D = m(X) + ν`

- "Partials out" X from both Y and D by computing residuals, then estimates θ as the regression of the outcome residual on the treatment residual.

**Formula:**

  `Ỹ = Y − ℓ(X)`        where `ℓ(X) = E[Y|X]` — the outcome nuisance function
  `D̃ = D − m(X)`        where `m(X) = E[D|X]` — the propensity/treatment nuisance function
  `θ̂ = Σᵢ D̃ᵢ·Ỹᵢ  /  Σᵢ D̃ᵢ²`

- **θ̂**: the causal parameter (ATE under the PLR model — assumed constant across units).
- **Ỹᵢ, D̃ᵢ**: "residual outcome" and "residual treatment" for unit i — the variation in Y and D *not explained* by X. Any remaining co-movement between them is attributed to the causal effect.
- **ℓ(X), m(X)**: the two "nuisance functions" — can be estimated with *any* flexible ML model (gradient boosting, random forest, etc.), not just linear regression.

**ATT-weighted version** (reweight the moment condition by the propensity score):

  `θ̂_ATT = Σᵢ e(Xᵢ)·D̃ᵢ·Ỹᵢ  /  Σᵢ e(Xᵢ)·D̃ᵢ²`

- **e(Xᵢ) = m(Xᵢ)**: the propensity score for unit i — up-weights units that "look like" treated units, shifting the estimand from the population average (ATE) toward the treated subpopulation (ATT).

**Cross-fitting:**
- Split the data into K folds. For each fold k, fit ℓ(X) and m(X) on the *other* K−1 folds, and predict (compute residuals) on fold k.
- Every observation's Ỹ and D̃ come from out-of-sample nuisance predictions — prevents the nuisance models from overfitting their own residuals, which would otherwise bias θ̂ toward zero.

**Neyman orthogonality:**
- The score function (the equation θ̂ solves) has the property that its derivative with respect to the nuisance functions, evaluated at their true values, is zero.
- **Practical consequence:** small/slow-converging errors in ℓ(X) and m(X) (which is what flexible ML models produce) affect θ̂ only at *second order* — θ̂ can still converge at the standard parametric (1/√n) rate even though the nuisance models converge more slowly.

**Sandwich (robust) standard error:**

  `SE(θ̂) = sqrt( E[D̃²·ε̃²] / E[D̃²]² / n )`

- **ε̃ = Ỹ − θ̂·D̃**: the score residual — what's left over after accounting for the estimated effect.
- Heteroskedasticity-robust by construction — no assumption of constant error variance.

**Assumptions:**
- Unconfoundedness and positivity given X (same as IPW/AIPW — DML doesn't relax these, it relaxes the *functional form* assumption on the nuisance models).
- The PLR model itself — a *constant* treatment effect θ (homogeneous effects). If effects are heterogeneous, PLR-DML recovers a particular weighted average of the underlying CATE function, not a free-standing "the" effect.
- Nuisance models converge fast enough for the orthogonality argument to apply (a technical condition, generally satisfied by standard ML models on moderate-to-large N).

**When to use:**
- Large N, continuous and/or skewed outcome where linear outcome models would be misspecified.
- Confounding relationships are nonlinear or involve interactions that a linear model would miss.
- You need valid statistical inference (standard errors, CIs) alongside a flexible model — DML's orthogonality is specifically what makes "ML model + valid CI" possible.

**When not to use:**
- Small N — cross-fitting splits the data further, and nuisance models may not have enough data per fold to estimate well, even before any overfitting concerns.
- Poor overlap — DML inherits the same overlap requirement as IPW/AIPW; with poor overlap, large N can produce a *false sense of precision* (the "precision trap" — sandwich SE shrinks with N even as the effective identification collapses).
- You actually care about *heterogeneous* effects, not a single pooled number — use CausalForest or a MetaLearner instead.

---

### Causal Forest / Generalized Random Forests (GRF)

**What it means:**
- Extends the DML idea to estimate a *function* τ(X) — the CATE — rather than a single number.
- Each tree in the forest splits the covariate space to maximize *heterogeneity in the estimated treatment effect* between resulting leaves (not heterogeneity in Y directly, as a standard regression tree would).
- For a target point x, the forest defines a set of weights αᵢ(x) — roughly, how often training observation i ends up in the same leaf as x across all trees — and τ(x) is estimated by solving a locally-weighted version of the DML moment equation.

**Formula:**

  `τ̂(x)` solves: `Σᵢ αᵢ(x)·ψ(Yᵢ, Dᵢ; τ, ν̂) = 0`

- **αᵢ(x)**: forest-based similarity weight — how relevant observation i is to estimating the effect at covariate value x. Observations that frequently share a leaf with x get higher weight.
- **ψ(·)**: the same Neyman-orthogonal moment function used in DML, but solved *locally* (per region of covariate space) rather than globally.
- **ν̂**: nuisance estimates (ℓ(X), m(X)) — typically estimated globally first, then used in the local moment equation.

**Aggregation to ATE / ATT:**

  `ATE = (1/n) Σᵢ τ̂(Xᵢ)`
  `ATT = (1/n₁) Σᵢ:Dᵢ=1 τ̂(Xᵢ)`        (average over treated units only)

**Honesty:**
- Each tree uses *separate* subsamples for (a) choosing where to split and (b) estimating τ within the resulting leaves.
- Prevents the tree from "overfitting" its own splits — i.e., from finding splits that look like they create heterogeneity just by chance, then estimating an inflated effect within those same chance-driven groups.
- This is what allows τ̂(x) to be asymptotically normal — enabling pointwise confidence intervals.

**Assumptions:**
- Same core unconfoundedness/positivity/SUTVA assumptions as DML.
- Enough observations *per region of covariate space* for forest leaves to contain a reasonable number of both treated and control units — CATE estimation is inherently more data-hungry than ATE estimation.

**When to use:**
- Large N and a reasonable hypothesis that treatment effects vary across the population.
- You want to identify *which* covariates drive effect heterogeneity (feature importance for heterogeneity) and/or produce a per-unit effect estimate for targeting.
- Pointwise inference (confidence intervals for τ(x) at specific x) is needed, not just a single ATE number.

**When not to use:**
- N is too small for forest leaves to contain enough observations of both treatment groups — CATE estimates become noisy and the "honesty" sample-splitting compounds the data-hunger problem.
- Note on inference: the *analytic* confidence interval for the aggregate ATE/ATT (derived from the forest's asymptotic variance formula) can be very wide — sometimes wider than a bootstrap CI for the same quantity — because it reflects leaf-level variance that partially cancels out in the aggregate but isn't fully accounted for analytically. When the analytic ATE CI looks implausibly wide, cross-check with a bootstrap CI before concluding the estimate itself is unreliable.

---

### MetaLearners: S-Learner, T-Learner, X-Learner

**What it means (shared idea):**
- A "meta" framework — take any off-the-shelf supervised ML model (regressor or classifier) as a building block, and combine predictions in different ways to estimate τ(x) = E[Y(1)−Y(0)|X=x].

**S-Learner ("Single"):**

  Fit one model: `μ̂(x,d) = E[Y | X=x, D=d]`  (D included as a feature alongside X)
  `τ̂_S(x) = μ̂(x,1) − μ̂(x,0)`

- **μ̂(x,d)**: a single model where the treatment indicator D is just another input feature.
- **Risk:** if D has a weak signal relative to the other features in X (common when treatment effects are small relative to covariate-driven variation in Y), many ML models will effectively "ignore" D — leading to τ̂_S(x) ≈ 0 for most x even when a real effect exists. This is a shrinkage-toward-zero bias, most severe for flexible/regularized models.

**T-Learner ("Two"):**

  Fit two separate models:
  `μ̂₁(x) = E[Y|X=x, D=1]`  — fit using only treated units
  `μ̂₀(x) = E[Y|X=x, D=0]`  — fit using only control units
  `τ̂_T(x) = μ̂₁(x) − μ̂₀(x)`

- **Risk:** each model is trained on only half the data (the treated-only or control-only subsample). If one group is much smaller than the other, that group's model has higher variance, and τ̂_T inherits that variance everywhere — even at covariate values where the smaller group has few observations.

**X-Learner ("Cross"):**

  Step 1 — fit μ̂₁(x), μ̂₀(x) as in T-Learner.
  Step 2 — compute *imputed individual treatment effects*:
  `D₁ᵢ = Yᵢ − μ̂₀(Xᵢ)`   for each treated unit i  (their actual outcome minus the *predicted* control outcome)
  `D₀ᵢ = μ̂₁(Xᵢ) − Yᵢ`   for each control unit i  (the *predicted* treated outcome minus their actual outcome)
  Step 3 — fit two more models on these imputed effects:
  `τ̂₁(x) = E[D₁|X=x]`  (fit on treated units' imputed effects)
  `τ̂₀(x) = E[D₀|X=x]`  (fit on control units' imputed effects)
  Step 4 — combine via a weighting function g(x) (often the propensity score or 1 − e(x)):
  `τ̂_X(x) = g(x)·τ̂₀(x) + (1−g(x))·τ̂₁(x)`

- **D₁ᵢ, D₀ᵢ**: "imputed" treatment effects — for a treated unit, we observe Y(1) and impute Y(0) via μ̂₀; for a control unit, we observe Y(0) and impute Y(1) via μ̂₁.
- **g(x)**: a weight that determines how much to trust τ̂₁ vs τ̂₀ at point x — typically weighted toward whichever group is *larger* (more reliable estimate) at that covariate value.
- **Key idea:** the larger group's model (e.g., μ̂₁ if treated >> control) is used to generate imputed effects *for the smaller group*, "borrowing strength" across groups — the smaller group's effect estimates benefit from the larger group's better-fit outcome model.

**Assumptions (all three):**
- Same unconfoundedness/positivity assumptions as other CATE methods — MetaLearners are a *computational framework*, not a new identification strategy.

**When to use:**
- **S-Learner**: small datasets, or when you have strong prior reason to expect a small/homogeneous effect and want a conservative (shrunk-toward-zero) estimate; simplest to implement.
- **T-Learner**: roughly balanced treated/control group sizes, and enough data in *both* groups to fit two separate flexible models well.
- **X-Learner**: imbalanced group sizes (one group substantially larger than the other) — the cross-imputation step specifically addresses this by letting the larger group's model inform the smaller group's effect estimates.

**When not to use:**
- **S-Learner**: when treatment is expected to have a strong, heterogeneous effect — shrinkage will mute exactly the heterogeneity you're trying to detect.
- **T-Learner**: when one group is small — that group's model will be noisy and the noise propagates into τ̂_T everywhere.
- **X-Learner**: when both groups are similarly sized *and* large — the extra cross-imputation machinery adds complexity without a corresponding benefit over T-Learner in that setting.

---

### DR-Learner and TMLE (brief)

**DR-Learner ("Doubly Robust Learner"):**
- A two-stage CATE estimator. Stage 1: fit μ̂₁(x), μ̂₀(x), and ê(x) (propensity) as in AIPW. Compute a "pseudo-outcome" for each unit:

  `φ̂ᵢ = [μ̂₁(Xᵢ) − μ̂₀(Xᵢ)] + Dᵢ·(Yᵢ−μ̂₁(Xᵢ))/ê(Xᵢ) − (1−Dᵢ)·(Yᵢ−μ̂₀(Xᵢ))/(1−ê(Xᵢ))`

- **φ̂ᵢ**: an AIPW-style "individual effect estimate" for unit i — combines the outcome-model-based effect with a propensity-weighted correction term, same logic as AIPW's doubly-robust property.
- Stage 2: regress φ̂ on X using any flexible model to get τ̂_DR(x).
- **When to use:** want a doubly-robust CATE estimator (protection if either the outcome or propensity model is misspecified) with a simple two-stage implementation.
- **When not to use:** when stage-1 nuisance estimates are poor — errors compound into the pseudo-outcome and then into the stage-2 regression; with very limited data, the two-stage structure may have more variance than a one-stage method like CausalForest.

**TMLE (Targeted Maximum Likelihood Estimation):**
- Starts from an initial outcome model, then applies a "targeting" step — a small parametric adjustment ("clever covariate" fluctuation) chosen specifically so the updated model satisfies the efficient influence function equation for the target estimand (e.g., ATE).
- Produces an estimator that is simultaneously doubly robust *and* asymptotically efficient (achieves the lowest possible variance among consistent estimators), with valid CIs from the start.
- **When to use:** when both doubly-robust consistency *and* asymptotic efficiency/valid inference are required — common in settings (e.g., clinical/epidemiological research) with strict requirements on estimator properties.
- **When not to use:** when a simpler estimator (AIPW, DML) already provides adequate robustness and the added implementation complexity of the targeting step isn't justified by the application.
