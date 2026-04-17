# Term paper plan — extending Aguirregabiria (2022) to quantity-dependent duration dependence

## Context

**Course.** Dynamic Programming & Structural Econometrics (UCPH), Part V — term paper. The paper must demonstrate the ability to model, solve/estimate, and analyze a dynamic problem. Implementation is required. Deadline: **June 3, 2026**; proposal due April 23; research workshop April 29–30.

**Source paper.** Aguirregabiria (2022, *Econometrics Journal*), "Dynamic demand for differentiated products with fixed-effects unobserved heterogeneity." The model: forward-looking consumers, multinomial brand choice + no-purchase, FE unobserved heterogeneity, dynamics from (i) last-brand dependence (habits/switching costs) and (ii) duration since last purchase (depletion/depreciation). The headline result is a **sufficient-statistic conditional maximum likelihood (CML) estimator** that differences out the continuation value, so no DP solution is needed during estimation.

**The gap we extend.** Aguirregabiria explicitly flags this in Section 5 (Conclusions):

> "The model assumes that consumers buy at most one unit of the product per period. However, it is well known that forward-looking consumers can buy for inventory (Hendel and Nevo, 2006, 2013). It would be interesting to extend the model to incorporate the possibility of consumers purchasing multiple units."

Our contribution: relax the unit-purchase assumption and let depletion depend **jointly on brand, quantity, and duration**: `β^dep(j, q, d)`. Under single-unit purchases, duration `d` is a sufficient summary of inventory age; under multi-unit purchases it is not — a consumer who bought 3 units faces different depletion dynamics than one who bought 1. This is both an economically meaningful extension and — because quantity is another endogenous state determined by choice — it is a non-trivial test of whether the CML sufficient-statistic machinery still works.

**Feasibility verdict: YES.** The extension is (a) explicitly invited by the author, (b) compatible in principle with the sufficient-statistic CML framework (quantity is just another endogenous state following a deterministic transition rule, same as last-brand `ℓ` and duration `d`), and (c) empirically tractable with standard consumer scanner data (Nielsen Homescan records both brand and quantity per trip).

---

## Extended model

### State variables
- `ℓ_{it}` — brand chosen at last purchase, `∈ {1,...,J}`
- `d_{it}` — duration since last purchase, `∈ {1,...,D*}`
- **`q_{it}` — quantity purchased at last purchase event, `∈ {1,...,Q̄}`** (new)

### Choice set
`y_{it} ∈ {0} ∪ {(j, q) : j ∈ {1,...,J}, q ∈ {1,...,Q̄}}`. Total alternatives: `J·Q̄ + 1`.

### Transition rule
```
(ℓ_{i,t+1}, d_{i,t+1}, q_{i,t+1}) = (ℓ_{it}, d_{it} + 1, q_{it})      if y_{it} = 0
                                  = (j, 1, q)                         if y_{it} = (j, q)
```

### Utility
```
U_{it} = 0 (no purchase):
    α_i(ℓ) + γ·h(μ_i) − β^dep(ℓ, q_{it}, d_{it}) + ε_{it}(0)

U_{it} = (j, q) (buy q units of brand j):
    α_i(j) + γ·h(μ_i − q·p_{it}(j)) − β^sc(ℓ_{it}, j) − β^qty(j, q) + ε_{it}(j,q)
```

Where:
- **`β^dep(j, q, d)`** is the non-parametric depletion function — the target object of interest. Saturation `β^dep(j, q, d) = β^dep(j, q, d*_{j,q})` for `d ≥ d*_{j,q}`, following Aguirregabiria's Assumption 3.2 extended pointwise in `q`.
- **`β^qty(j, q)`** captures any static disutility/convenience of buying multiple units at once (bulk handling cost), not driven by dynamics. Normalized `β^qty(j, 1) = 0`.
- Budget effect `q·p_{it}(j)` correctly enters `h(·)` — buying more units is more expensive.
- Switching cost `β^sc(ℓ, j)` depends on brand identity only, not quantity (standard).

### Parameter caveat on "fully nonparametric β^dep"
You chose the fully nonparametric specification. With `J = 3`, `Q̄ = 3`, `d* = 4`, this is already 36 depletion parameters. Point identification via CML requires enough history pairs for each `(j, q, d)` cell. The plan below therefore includes:
1. A **Monte Carlo sample-size diagnostic** that reports which `(j,q,d)` cells are identified at a given `N, T`.
2. A fallback **shape-restricted sieve** (monotone in `d`, monotone in `q`, subject to saturation) as a robustness variant, to be reported alongside the unrestricted estimates.

This is the honest way to deliver "fully nonparametric" without overstating identification.

---

## Identification strategy

Aguirregabiria's CML identification (Section 3) rests on finding **history pairs `A` and `B` with identical sufficient statistics `s(A) = s(B)`** but differing in the parameter of interest via `c(A) − c(B)`. Then `θ_k = log P(A) − log P(B)`, which differences out the incidental parameters `α_i` and — crucially — the continuation value `v_{α_i}(·)`.

### New pairs for `β^dep(j, q, d)`
Extend Aguirregabiria's duration-identifying pair (Eq. 3.25, p. 18):

- `A_{j,q,n} = ((j,q), 0_{n−1}, (j,q), 0_{n+1})`
- `B_{j,q,n} = ((j,q), 0_n,   (j,q), 0_n)`

Both histories purchase the same `(j, q)` twice and spend the same total time not purchasing. They visit different `(ℓ, q, d)` tuples at the σ-points, but under saturation `d ≥ d*_{j,q}` the continuation values match. Then (analogous to Aguirregabiria's Prop., Eq. 3.31):

```
log P(A_{j,q,n}) − log P(B_{j,q,n}) = −β^dep(j, q, d*_{j,q})     for n = d*_{j,q} − 1
```

Varying `q` across history pairs identifies the quantity-dependent depletion rate.

### Identification of `β^qty(j, q)`
From history pairs differing only in a single purchase event's quantity, at a time when subsequent dynamics are matched. Requires price variation in the transitory component `e_t` (as in Aguirregabiria's γ identification) — the permanent price component `z_t` must be held constant across the two compared sub-histories, transitory `e_t` allowed to vary to trace out quantity-price responsiveness.

### Identification of `γ` and `β^sc`
The original Aguirregabiria pairs (Eqs. 3.16 and 3.21) continue to work, restricted to sub-histories where `q_{it}` is constant. This sub-case preserves the original result.

### What this buys us
- **No DP solution required** during estimation — the CML property survives.
- State space blow-up (`J·Q̄ + 1` choices, three-dimensional endogenous state) is irrelevant for estimation cost.
- Fixed-effects `α_i` remain fully unrestricted.

### Cost / risk
- More parameters → need more observations per `(j, q, d)` cell → finite-sample performance will be noisier than the original paper's.
- Saturation point `d*_{j,q}` must be identified jointly with the depletion parameters (Aguirregabiria's Eq. 3.30 extends pointwise).

---

## Deliverables (structure of the term paper, ~25 pp)

1. **Introduction** — motivation, contribution, connection to Hendel & Nevo (2006).
2. **Model** — extended utility, transition rule, consumer problem.
3. **Identification** — formal propositions for `β^dep(j,q,d)`, `β^qty`, and unchanged identification of `γ`, `β^sc`.
4. **Estimation** — kernel-weighted CML extended to the new state space.
5. **Monte Carlo** — DGP, sample size, parameter recovery, comparison vs. misspecified (quantity-ignorant) estimator.
6. **Empirical application** — Nielsen Homescan (laundry detergent or similar storable), CML estimates, demand elasticity counterfactuals.
7. **Conclusion** — limitations, future work (multi-unit inventory carryover; endogenous quantity choice at the firm side).

---

## Implementation plan (Python)

### Phase 0 — Setup (Week 1)
- Set up a Python project (`poetry` or `uv`): `numpy`, `scipy`, `pandas`, `numba` (speed), `matplotlib`, `statsmodels` for comparisons.
- Create directory: `term_paper/` under the course repo.

### Phase 1 — Analytical derivation (Weeks 1–2)
- Write out extended utility, CCP, log-CCP decomposition (analog of Aguirregabiria Eq. 3.4).
- Derive the new history pairs and sufficient-statistic vectors `s_i`, `c_i` (analog of Eqs. 3.7–3.8).
- Prove identification propositions; write the proofs up as an appendix.

### Phase 2 — Simulator (Week 2)
Build a forward-looking DGP that actually solves the consumer's Bellman problem:
- `simulate_dgp(N, T, params, seed)` — solves the DP via value function iteration on the `(ℓ, d, q, z, e)` state space, draws ε's, generates panel `{y_it, p_it}`.
- This is *only* needed to generate data; the estimator never solves the DP. This dual structure is the key pedagogical point of the paper — showing that CML bypasses the curse of dimensionality that the DGP suffers from.
- Reference: Part I of the course for VFI implementation.

### Phase 3 — CML estimator (Weeks 3–4)
- `build_history_pairs(data, M)` — enumerate the `M` identifying (A, B) pairs.
- `conditional_loglik(θ, data, pairs, kernel_bw)` — kernel-weighted CML objective (Eq. 4.4).
- Maximize with `scipy.optimize.minimize` (BFGS or trust-ncg); log-likelihood is globally concave in `θ` (per Aguirregabiria Eq. 3.13), so any local optimizer works.
- Sandwich variance estimator via numerical Hessian + outer-product score.

### Phase 4 — Monte Carlo (Week 4)
- 500 replications at `N ∈ {1000, 5000, 10000}`, `T ∈ {8, 12, 16}`.
- Report: bias, RMSE, coverage of 95% CIs, and **identification diagnostic** (which `(j,q,d)` cells have non-degenerate score contributions).
- Comparator 1: misspecified estimator that ignores `q` (the original Aguirregabiria estimator applied to multi-unit data) — should show bias.
- Comparator 2: shape-restricted sieve — should show variance reduction.

### Phase 5 — Empirical application (Week 5)
- **Data.** Nielsen Homescan / Kilts Consumer Panel (UCPH has access via CBS; confirm early). Storable category: liquid laundry detergent (Tide, Persil, Gain — mirrors Aguirregabiria's Fig. 1).
- **Sample.** ~2,000 households, 2010–2013, households with ≥ 4 purchase events.
- **Estimate** the extended model; report depletion-by-quantity curves.
- **Counterfactual.** Compare implied short-run vs. long-run price elasticity when the analyst ignores quantity.

### Phase 6 — Writing (Week 6, overlapping)
Draft in LaTeX (use `econometrica.cls` or a neutral template). Figures generated from Python via `matplotlib` / `tikzplotlib`.

---

## Critical files / references to reuse

- **Source paper PDF:** `/Users/olivereiberg/Documents/Economics/MA/2_Semester/Dynamic_Programming/dynamic_demand_fixed_effects.pdf` — Sections 2 (model), 3 (identification), 4 (estimation) are the direct templates.
- **Companion paper:** Aguirregabiria, Gu, Luo (2021, *J. Econometrics*) — "Sufficient statistics for unobserved heterogeneity in dynamic structural logit models." Download and read; it gives the general theorem this paper applies.
- **Course materials:** Part I (VFI) for the simulator, Part II (NFXP/CCP) for estimator comparison context, Part III (EGM) not directly relevant.
- **Data docs:** Nielsen Kilts portal at University of Chicago Booth (the paper cites Nielsen/NielsenIQ, 2022).

---

## Verification / success criteria

- **Analytical:** identification propositions stated and proven for `β^dep(j,q,d)`, `β^qty(j,q)`; standard `γ`, `β^sc` identification recovered as a corollary.
- **Monte Carlo:**
  - Extended CML estimator unbiased at `N ≥ 5000, T ≥ 12` for parameters whose identifying cells have ≥ 50 observations.
  - Misspecified (quantity-ignorant) estimator shows statistically significant bias in `β^dep` when the true DGP has `Q̄ > 1`.
  - 95% CI coverage within [0.92, 0.97].
- **Empirical:** CML converges on the Homescan sample; estimated depletion curves are monotone in `d` and scale sensibly with `q` (even without imposing the shape restriction).
- **End-to-end test.** A single script `run_paper.py` that (a) simulates, (b) estimates on simulated data, (c) runs one Monte Carlo replication, (d) produces all paper figures from cached Homescan estimates. Runs in under 30 minutes on a laptop.

---

## Open risks and mitigations

| Risk | Mitigation |
|---|---|
| Nielsen data access delayed | Start Phase 5 with a public scanner dataset (IRI Academic, Dominick's); run the full paper on simulated data only if needed. |
| Nonparametric `β^dep(j,q,d)` under-identified at student-feasible sample sizes | Shape-restricted sieve as primary result; unrestricted as robustness. |
| Derivation bugs in history pairs (wrong σ-matching) | Validate by simulating from the DGP and checking the empirical `log P(A) − log P(B)` matches the theoretical parameter identity before running estimation. This is the single most important sanity check. |
| Scope creep | Hard-cap the empirical application at one product category. Counterfactuals limited to price elasticity; do not endogenize firm pricing. |

---

## Immediate next steps (this week)

1. Confirm Nielsen/Kilts data access via UCPH channel; if unavailable, identify a public fallback.
2. Read Aguirregabiria, Gu & Luo (2021) end-to-end — the general sufficient-statistic theorem is load-bearing.
3. Draft proposal (due **April 23**): 2 pages, state the extension, sketch one identification result, commit to simulation + Nielsen application.
4. Start Phase 2 simulator in parallel — a working DGP unblocks everything downstream.
