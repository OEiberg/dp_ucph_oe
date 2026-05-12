"""
nfxp_agu2022.py
===============
Monte Carlo simulation and NFXP estimation of the dynamic discrete-choice
demand model in Aguirregabiria (2022), with J=2 brands, T=52 weekly periods
(one calendar year), and homogeneous consumers (no unobserved heterogeneity).

Reference
---------
Aguirregabiria, V. (2022). Dynamic demand for differentiated products with
fixed-effects unobserved heterogeneity. *Econometrics Journal*.

Model summary
-------------
Every period t a consumer chooses y_it ∈ {0,1,2}:
    y=0  → no purchase (keeps consuming last brand, which depreciates)
    y=j  → buys brand j

Endogenous state x_it = (ell_it, d_it):
    ell_it  last brand purchased  ∈ {1,...,J}
    d_it    duration since last purchase, capped at D_MAX

Prices follow a Hi-Lo process (Assumption 2.1):
    p_t(j) = z(j) − disc(j)·e_t(j)
where z(j) is the persistent (regular) price and e_t(j) ∈ {0,1} is a
transitory promotion indicator that follows an independent two-state Markov
chain across brands.

Deterministic per-period utility (equation 2.6):
    y=0:  u = alpha(ell) + gamma·h(mu)          − beta_dep(ell)·d
    y=j:  u = alpha(j)   + gamma·h(mu − p_t(j)) − beta_sc(ell, j)

where h(c) = log(c) is the log-utility transformation.

Unobservables ε_it(j) are i.i.d. Type-I Extreme Value → multinomial logit.

Estimator: NFXP (Nested Fixed Point, Rust 1987)
----------------------------------------------
Outer loop:  Nelder-Mead optimizer searches over theta
Inner loop:  Value-function iteration (VFI) solves the Bellman equation
             at each candidate theta, yielding CCPs used to evaluate the
             full-panel log-likelihood.

Because there is no unobserved heterogeneity here, the full MLE identifies
all parameters including brand intercepts alpha(j), unlike the CML approach
which differences them out.

Parameter vector
----------------
theta = [alpha_2, gamma, beta_sc_12, beta_sc_21, beta_dep_1, beta_dep_2]
    alpha_1 = 0 is the normalisation (brand 1 is the reference brand).
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRIMITIVES  (change J, N, T, D_MAX here to experiment)
# ─────────────────────────────────────────────────────────────────────────────

J     = 2      # number of brands
T     = 52     # periods per consumer (one calendar year of weekly data)
N     = 2_000  # consumers per simulated panel (keep manageable for NFXP)
D_MAX = 2      # duration cap: Assumption 3.2; keeps state space small
DELTA = 0.95   # discount factor (calibrated, not estimated)
MU    = 15.0   # consumer disposable income (weekly)

# ── True DGP parameters ──────────────────────────────────────────────────────
# alpha(1)=0 by normalisation; we only set alpha(2) here.
ALPHA_TRUE    = np.array([0.0,  0.30])   # brand intercepts
GAMMA_TRUE    = 0.50                     # price sensitivity (marginal utility of income)
BETA_SC_TRUE  = np.array([[0.00, 0.55],  # beta_sc[k-1,j-1]: cost of switching k→j
                           [0.50, 0.00]])
BETA_DEP_TRUE = np.array([0.25, 0.25])   # depreciation rate per brand

# ── Hi-Lo price process (Assumption 2.1) ─────────────────────────────────────
BASE_PRICES   = np.array([10.0, 15.0])   # persistent price component z(j)
PROMO_DISC    = np.array([ 2.0,  2.0])   # discount when brand is on promotion
PROMO_ENTRY   = 0.18   # prob(promotion starts | currently not promoted)
PROMO_PERSIST = 0.62   # prob(promotion continues | currently promoted)

# ── Monte Carlo settings ─────────────────────────────────────────────────────
MC_REPS = 20    # number of independent replications
MC_SEED = 2024  # master seed for reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRICE / PROMOTION PROCESS
# ─────────────────────────────────────────────────────────────────────────────

# All 2^J binary promotion vectors: promo_states[s] = (e_1, e_2, ...)
promo_states = np.array(
    [[(s >> j) & 1 for j in range(J)] for s in range(2 ** J)], dtype=int
)
N_PROMO = len(promo_states)   # 4 for J=2


def make_promo_transition() -> np.ndarray:
    """
    Build the (N_PROMO × N_PROMO) Markov transition matrix for promotion
    states.  Brands' promotions are independent two-state Markov chains, so
    the joint transition probability is the product of the marginal ones
    (Assumption 2.1 A–B).
    """
    trans = np.empty((N_PROMO, N_PROMO))
    for s, curr in enumerate(promo_states):
        # Probability each brand is on promotion next period
        prob_on = np.where(curr == 1, PROMO_PERSIST, PROMO_ENTRY)
        for sp, nxt in enumerate(promo_states):
            # Joint probability = product over brands (independence)
            trans[s, sp] = np.prod(np.where(nxt == 1, prob_on, 1.0 - prob_on))
    return trans


PROMO_TRANS = make_promo_transition()   # shape (N_PROMO, N_PROMO)


def prices_from_state(e_idx: int) -> np.ndarray:
    """Return the (J,) price vector at promotion-state index e_idx."""
    return BASE_PRICES - PROMO_DISC * promo_states[e_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  UTILITY FUNCTION  (equation 2.6)
# ─────────────────────────────────────────────────────────────────────────────

def h(c: np.ndarray) -> np.ndarray:
    """Log-utility from net resources; small floor prevents log(0)."""
    return np.log(np.maximum(c, 1e-8))


def flow_util(choice, last_brand, duration, e_idx,
              alpha, gamma, beta_sc, beta_dep) -> float:
    """
    Deterministic part of period-utility for one observation (equation 2.6).

    Parameters
    ----------
    choice     : int  — 0 = no purchase, j∈{1,...,J} = buy brand j
    last_brand : int  — ell ∈ {1,...,J}
    duration   : int  — d ∈ {0,...,D_MAX}
    e_idx      : int  — index into promo_states
    alpha      : (J,) brand intercepts
    gamma      : float price sensitivity
    beta_sc    : (J,J) switching cost matrix, beta_sc[k-1,j-1]
    beta_dep   : (J,) depreciation rates per brand
    """
    l = last_brand - 1          # convert to 0-indexed
    prices = prices_from_state(e_idx)

    if choice == 0:
        # No purchase: consumer keeps last brand; depreciation accumulates
        return alpha[l] + gamma * h(MU) - beta_dep[l] * duration

    j = choice - 1              # convert to 0-indexed
    # Purchase brand j; switching cost applies if j ≠ last_brand
    return alpha[j] + gamma * h(MU - prices[j]) - beta_sc[l, j]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INNER LOOP: VALUE FUNCTION ITERATION  (Bellman fixed point)
# ─────────────────────────────────────────────────────────────────────────────

def solve_vfi(alpha, gamma, beta_sc, beta_dep,
              tol: float = 1e-10, max_iter: int = 2_000) -> np.ndarray:
    """
    Solve the consumer's infinite-horizon Bellman equation by VFI.

    This is the *inner loop* of NFXP: given a candidate parameter vector,
    iterate the Bellman operator until the value function converges.

    With i.i.d. Type-I Extreme Value shocks the Bellman operator is:

        T(V)[l, d, e] = log Σ_j exp( u(j,l,d,e) + δ·EV(j,l,d,e) )

    where EV(j,l,d,e) = Σ_{e'} P(e'|e)·V( l'(j,l,d), d'(j,l,d), e' )
    and (l'(j,l,d), d'(j,l,d)) is the deterministic next state (eq. 2.1).

    Parameters
    ----------
    alpha, gamma, beta_sc, beta_dep : structural parameters at current theta
    tol      : sup-norm convergence tolerance
    max_iter : safety cap on iterations

    Returns
    -------
    V : ndarray, shape (J, D_MAX+1, N_PROMO) — ex-ante value function
    """
    V = np.zeros((J, D_MAX + 1, N_PROMO))

    for _ in range(max_iter):
        # ── Expected continuation value ──────────────────────────────────────
        # Integrate V over next period's promotion state using the Markov
        # transition matrix.  Shape: (J, D_MAX+1, N_PROMO).
        EV = (V.reshape(J * (D_MAX + 1), N_PROMO) @ PROMO_TRANS.T
              ).reshape(J, D_MAX + 1, N_PROMO)

        V_new = np.empty_like(V)

        for l_idx in range(J):
            ell = l_idx + 1
            for d in range(D_MAX + 1):
                d_next = min(d + 1, D_MAX)   # duration if no purchase
                for e in range(N_PROMO):
                    # Choice-specific values Q(j) = u(j) + δ·EV(next state)
                    Q = np.empty(J + 1)

                    # j=0: no purchase → state stays (ell, d_next)
                    Q[0] = (flow_util(0, ell, d, e, alpha, gamma, beta_sc, beta_dep)
                            + DELTA * EV[l_idx, d_next, e])

                    # j>0: buy brand j → state resets to (j, 0)
                    for j_idx in range(J):
                        Q[j_idx + 1] = (
                            flow_util(j_idx + 1, ell, d, e, alpha, gamma, beta_sc, beta_dep)
                            + DELTA * EV[j_idx, 0, e]
                        )

                    # Log-sum-exp = E[max(Q + ε)] for EV1 shocks (closed form)
                    q_max = Q.max()
                    V_new[l_idx, d, e] = q_max + np.log(np.exp(Q - q_max).sum())

        # Convergence check (sup-norm on value function)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new

        V = V_new

    # Return best iterate if max_iter is reached without convergence
    return V


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONDITIONAL CHOICE PROBABILITIES  (from solved V)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ccps(V, alpha, gamma, beta_sc, beta_dep) -> np.ndarray:
    """
    Compute the (J, D_MAX+1, N_PROMO, J+1) array of conditional choice
    probabilities implied by the solved value function V.

    The CCPs are multinomial logit softmax over the choice-specific values
    Q(j) = u(j,l,d,e) + δ·EV(next state | j,l,d,e).
    """
    EV = (V.reshape(J * (D_MAX + 1), N_PROMO) @ PROMO_TRANS.T
          ).reshape(J, D_MAX + 1, N_PROMO)

    P = np.empty((J, D_MAX + 1, N_PROMO, J + 1))

    for l_idx in range(J):
        ell = l_idx + 1
        for d in range(D_MAX + 1):
            d_next = min(d + 1, D_MAX)
            for e in range(N_PROMO):
                Q = np.empty(J + 1)
                Q[0] = (flow_util(0, ell, d, e, alpha, gamma, beta_sc, beta_dep)
                        + DELTA * EV[l_idx, d_next, e])
                for j_idx in range(J):
                    Q[j_idx + 1] = (
                        flow_util(j_idx + 1, ell, d, e, alpha, gamma, beta_sc, beta_dep)
                        + DELTA * EV[j_idx, 0, e]
                    )
                # Softmax (numerically stable via max subtraction)
                q_shifted = Q - Q.max()
                w = np.exp(q_shifted)
                P[l_idx, d, e, :] = w / w.sum()

    return P


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _sample_rows(rng, row_probs: np.ndarray) -> np.ndarray:
    """
    Vectorised categorical draw: for each row i in row_probs (shape N×K),
    draw one index from the corresponding probability vector.

    Uses the inverse-CDF trick:
        k* = number of cumsum entries that are strictly less than u_i
    which gives the unique k with CDF[k-1] < u_i ≤ CDF[k].
    """
    u = rng.random(len(row_probs))          # (N,) uniform draws
    cumsum = np.cumsum(row_probs, axis=1)   # (N, K) running totals
    return (u[:, None] > cumsum).sum(axis=1)


def simulate_panel(P_true: np.ndarray,
                   n_consumers: int = N,
                   n_periods: int = T,
                   seed=None) -> dict:
    """
    Simulate a consumer panel from the model given true CCPs P_true.

    Parameters
    ----------
    P_true     : (J, D_MAX+1, N_PROMO, J+1) array of true CCPs
    n_consumers, n_periods : panel dimensions
    seed       : int or None; passed to np.random.default_rng

    Returns
    -------
    dict with four (n_consumers × n_periods) integer arrays:
        Y     — observed choice in {0,...,J}
        L     — state ell (last brand) at the start of period t
        D     — state d (duration) at the start of period t
        E_IDX — promotion state index at period t
    """
    rng = np.random.default_rng(seed)

    Y     = np.zeros((n_consumers, n_periods), dtype=int)
    L     = np.zeros((n_consumers, n_periods), dtype=int)
    D     = np.zeros((n_consumers, n_periods), dtype=int)
    E_IDX = np.zeros((n_consumers, n_periods), dtype=int)

    # Initialise consumer states randomly
    ell   = rng.integers(1, J + 1, size=n_consumers)          # random starting brand
    dur   = rng.integers(0, D_MAX + 1, size=n_consumers)      # random starting duration
    e_idx = rng.integers(0, N_PROMO, size=n_consumers)        # random starting promo state

    for t in range(n_periods):
        # Record beginning-of-period states
        L[:, t]     = ell
        D[:, t]     = dur
        E_IDX[:, t] = e_idx

        # Look up each consumer's CCP row and draw a choice
        probs = P_true[ell - 1, np.minimum(dur, D_MAX), e_idx, :]   # (N, J+1)
        y = _sample_rows(rng, probs)
        Y[:, t] = y

        # Update endogenous states (transition rule, equation 2.1)
        bought = y > 0
        ell = np.where(bought, y, ell)                              # new last brand
        dur = np.where(bought, 0, np.minimum(dur + 1, D_MAX))       # reset or increment

        # Update exogenous promotion state using the Markov chain
        # PROMO_TRANS[e_idx] gives each consumer's transition row
        e_idx = _sample_rows(rng, PROMO_TRANS[e_idx])

    return {"Y": Y, "L": L, "D": D, "E_IDX": E_IDX}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LOG-LIKELIHOOD  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood(data: dict, P: np.ndarray) -> float:
    """
    Compute the sum of log-likelihoods over all (consumer, period) pairs.

    For each observation (i,t):
        contribution = log P( y_it | ell_it, d_it, e_it; theta )

    Uses NumPy advanced indexing to avoid Python loops over observations.

    Parameters
    ----------
    data : panel dict from simulate_panel
    P    : (J, D_MAX+1, N_PROMO, J+1) CCP array from compute_ccps

    Returns
    -------
    float — sum of log-likelihoods (higher is better)
    """
    Y, L, D, E = data["Y"], data["L"], data["D"], data["E_IDX"]

    # Advanced indexing: pick out P[l_idx, d, e, y] for every (i,t)
    probs = P[L - 1, D, E, Y]                   # shape (N, T)

    # Clip at a tiny positive value to prevent log(0) if a state is never visited
    return float(np.sum(np.log(np.maximum(probs, 1e-300))))


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PARAMETER (UN)PACKING
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["alpha_2", "gamma", "beta_sc_12", "beta_sc_21",
               "beta_dep_1", "beta_dep_2"]

# True parameter vector in theta-format (used as DGP and as reference in MC)
THETA_TRUE = np.array([
    ALPHA_TRUE[1],          # alpha_2  (alpha_1 = 0 by normalisation)
    GAMMA_TRUE,             # gamma
    BETA_SC_TRUE[0, 1],     # beta_sc(1→2)
    BETA_SC_TRUE[1, 0],     # beta_sc(2→1)
    BETA_DEP_TRUE[0],       # beta_dep brand 1
    BETA_DEP_TRUE[1],       # beta_dep brand 2
])


def unpack(theta: np.ndarray):
    """
    Unpack the 6-element theta vector into named parameter arrays.

    theta = [alpha_2, gamma, beta_sc_12, beta_sc_21, beta_dep_1, beta_dep_2]

    Returns (alpha, gamma, beta_sc, beta_dep).
    """
    alpha    = np.array([0.0, theta[0]])         # alpha_1 = 0 normalisation
    gamma    = float(theta[1])
    beta_sc  = np.array([[0.0,    theta[2]],     # from brand 1: to brand 2
                          [theta[3], 0.0   ]])    # from brand 2: to brand 1
    beta_dep = np.array([theta[4], theta[5]])
    return alpha, gamma, beta_sc, beta_dep


# ─────────────────────────────────────────────────────────────────────────────
# 9.  NFXP OBJECTIVE  (outer loop)
# ─────────────────────────────────────────────────────────────────────────────

def nfxp_objective(theta: np.ndarray, data: dict) -> float:
    """
    NFXP objective: negative log-likelihood as a function of theta.

    Algorithm
    ---------
    1. Unpack theta into named structural parameters.
    2. *Inner loop*: run VFI to convergence → value function V*(theta).
    3. Compute CCPs from V*(theta).
    4. Evaluate the log-likelihood on the observed data.

    The outer optimizer (Nelder-Mead) minimises this function.

    Parameters
    ----------
    theta : (6,) parameter vector  [alpha_2, gamma, β_sc12, β_sc21, βd1, βd2]
    data  : panel dict from simulate_panel

    Returns
    -------
    float — negative log-likelihood (minimised by the outer optimizer)
    """
    alpha, gamma, beta_sc, beta_dep = unpack(theta)

    # ── Inner loop: solve Bellman equation at current theta ──────────────────
    V = solve_vfi(alpha, gamma, beta_sc, beta_dep)

    # ── Compute CCPs and evaluate likelihood ─────────────────────────────────
    P = compute_ccps(V, alpha, gamma, beta_sc, beta_dep)
    return -log_likelihood(data, P)


# ─────────────────────────────────────────────────────────────────────────────
# 10. NFXP ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def estimate_nfxp(data: dict,
                  theta0: np.ndarray = None,
                  verbose: bool = False):
    """
    Estimate structural parameters by NFXP using Nelder-Mead.

    Nelder-Mead (gradient-free simplex method) is used because:
      - The inner VFI introduces small numerical errors that can mislead
        finite-difference gradient approximations.
      - The state space is small (6 parameters), so Nelder-Mead is tractable.

    For a gradient-based alternative, L-BFGS-B with numerical gradients can
    be faster when the state space is larger.

    Parameters
    ----------
    data   : panel dict from simulate_panel
    theta0 : (6,) starting values; defaults to a vector of small positive numbers
    verbose: bool; passed to scipy's 'disp' option

    Returns
    -------
    OptimizeResult from scipy.optimize.minimize
    """
    if theta0 is None:
        # Safe default starting values (positive, away from boundaries)
        theta0 = np.array([0.1, 0.3, 0.3, 0.3, 0.1, 0.1])

    return minimize(
        fun=nfxp_objective,
        x0=theta0,
        args=(data,),
        method="Nelder-Mead",
        options={
            "maxiter": 10_000,
            "xatol":   1e-5,     # tolerance on parameter change
            "fatol":   1e-5,     # tolerance on function value change
            "disp":    verbose,
            "adaptive": True,    # adaptive simplex scales better to the dim
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(n_reps: int = MC_REPS, seed: int = MC_SEED):
    """
    Run the full Monte Carlo experiment.

    For each replication:
      1. Simulate a new panel from the true DGP.
      2. Estimate theta by NFXP.
      3. Record estimates, bias, and squared errors.

    Parameters
    ----------
    n_reps : number of independent replications
    seed   : master seed; each replication gets its own derived seed

    Returns
    -------
    results_df : DataFrame with one row per (replication × parameter)
    summary_df : DataFrame with mean bias, std dev, RMSE per parameter
    """
    rng_master = np.random.default_rng(seed)
    rep_seeds  = rng_master.integers(0, 1_000_000, size=n_reps)

    # Solve the DP once at the true parameters to get DGP choice probabilities.
    # All replications use the same P_true; randomness comes from the draw.
    alpha0, g0, sc0, dep0 = unpack(THETA_TRUE)
    V_true  = solve_vfi(alpha0, g0, sc0, dep0)
    P_true  = compute_ccps(V_true, alpha0, g0, sc0, dep0)

    print(f"\nNFXP Monte Carlo  |  J={J}, T={T}, N={N}, D_MAX={D_MAX}, "
          f"delta={DELTA}, reps={n_reps}")
    print(f"True theta: {dict(zip(PARAM_NAMES, THETA_TRUE))}\n")

    rows = []

    for rep in range(1, n_reps + 1):
        # ── Simulate panel for this replication ──────────────────────────────
        data = simulate_panel(P_true, n_consumers=N, n_periods=T,
                              seed=int(rep_seeds[rep - 1]))

        # ── Starting values: perturb truth slightly for robustness testing ───
        # Using a replication-specific perturbation keeps the MC reproducible.
        rng_start = np.random.default_rng(int(rep_seeds[rep - 1]) + 999)
        theta0 = THETA_TRUE + rng_start.normal(0.0, 0.05, size=len(THETA_TRUE))

        # ── Estimate ─────────────────────────────────────────────────────────
        t0     = time.perf_counter()
        result = estimate_nfxp(data, theta0=theta0)
        t_sec  = time.perf_counter() - t0

        print(f"  Rep {rep:>3}/{n_reps}  |  "
              f"converged={result.success}  nit={result.nit:>5}  "
              f"time={t_sec:>5.1f}s")

        # ── Store results for each parameter ─────────────────────────────────
        for k, name in enumerate(PARAM_NAMES):
            rows.append({
                "replication": rep,
                "parameter":   name,
                "true":        THETA_TRUE[k],
                "estimate":    result.x[k],
                "bias":        result.x[k] - THETA_TRUE[k],
                "sq_error":    (result.x[k] - THETA_TRUE[k]) ** 2,
                "converged":   int(result.success),
                "n_iter":      result.nit,
                "time_sec":    t_sec,
            })

    results_df = pd.DataFrame(rows)

    # ── Summary statistics ────────────────────────────────────────────────────
    summary_rows = []
    for name in PARAM_NAMES:
        sub = results_df[results_df["parameter"] == name]
        est = sub["estimate"].to_numpy()
        summary_rows.append({
            "parameter": name,
            "true":      THETA_TRUE[PARAM_NAMES.index(name)],
            "mean_est":  est.mean(),
            "bias":      sub["bias"].mean(),          # mean bias (ideally ≈ 0)
            "std_dev":   est.std(ddof=1),             # MC standard deviation
            "rmse":      np.sqrt(sub["sq_error"].mean()),  # root mean squared error
            "conv_rate": sub["converged"].mean(),     # fraction of converged runs
        })
    summary_df = pd.DataFrame(summary_rows)

    return results_df, summary_df


# ─────────────────────────────────────────────────────────────────────────────
# 12. DIAGNOSTICS  (single-panel pilot)
# ─────────────────────────────────────────────────────────────────────────────

def run_pilot(seed: int = 42, verbose: bool = True):
    """
    Single-panel diagnostic: simulate one dataset, estimate, and print results.

    Useful to check that VFI converges, the simulation looks reasonable, and
    the estimator recovers the true parameters before running the full MC.

    Parameters
    ----------
    seed    : random seed for the simulated panel
    verbose : print detailed output

    Returns
    -------
    result : OptimizeResult
    data   : simulated panel dict
    """
    if verbose:
        print("=" * 65)
        print("NFXP Pilot — Aguirregabiria (2022), J=2, homogeneous")
        print("=" * 65)

    # Step 1: Solve the DP at the true parameters
    if verbose:
        print("\n[1] Solving DP at true parameters (VFI)...")
    alpha0, g0, sc0, dep0 = unpack(THETA_TRUE)
    V_true = solve_vfi(alpha0, g0, sc0, dep0)
    P_true = compute_ccps(V_true, alpha0, g0, sc0, dep0)

    # Step 2: Simulate one panel
    if verbose:
        print("[2] Simulating panel...")
    data = simulate_panel(P_true, seed=seed)
    purch_rate = (data["Y"] > 0).mean()
    brand_shares = np.array([(data["Y"] == j).mean() for j in range(J + 1)])
    if verbose:
        print(f"     Purchase rate : {purch_rate:.1%}")
        print(f"     Brand shares  : no-purch={brand_shares[0]:.2f}  "
              + "  ".join(f"brand{j}={brand_shares[j]:.2f}" for j in range(1, J + 1)))

    # Step 3: Estimate by NFXP
    if verbose:
        print("[3] Estimating by NFXP (Nelder-Mead)...")
    t0 = time.perf_counter()
    result = estimate_nfxp(data, theta0=THETA_TRUE + 0.05, verbose=verbose)
    t_sec = time.perf_counter() - t0

    if verbose:
        print(f"\n     Time      : {t_sec:.1f}s")
        print(f"     Converged : {result.success}  (nit={result.nit})")
        print(f"\n     {'Parameter':<14}  {'True':>8}  {'NFXP':>8}  {'Bias':>8}")
        print("     " + "-" * 44)
        for k, name in enumerate(PARAM_NAMES):
            print(f"     {name:<14}  {THETA_TRUE[k]:>8.4f}  "
                  f"{result.x[k]:>8.4f}  {result.x[k] - THETA_TRUE[k]:>+8.4f}")

    return result, data


# ─────────────────────────────────────────────────────────────────────────────
# 13. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Pilot: single-panel estimate ─────────────────────────────────────────
    pilot_result, pilot_data = run_pilot(seed=42, verbose=True)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    results_df, summary_df = run_monte_carlo(n_reps=MC_REPS, seed=MC_SEED)

    print("\n" + "=" * 65)
    print("Monte Carlo Summary")
    print("=" * 65)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 65)

    # Optional: save results to CSV for further analysis
    results_df.to_csv("nfxp_mc_results.csv", index=False)
    summary_df.to_csv("nfxp_mc_summary.csv", index=False)
    print("\nResults saved to nfxp_mc_results.csv and nfxp_mc_summary.csv")
