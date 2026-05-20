"""
Monte Carlo Option Pricing under the Black-Scholes Framework
=============================================================

We price European and arithmetic Asian options by simulating asset price
paths under the risk-neutral measure via Euler-Maruyama discretisation of
Geometric Brownian Motion. To improve the efficiency of our estimator, we
apply the method of antithetic variates, which exploits negative correlation
between paired sample paths to reduce simulation variance. We verify the
European prices against the Black-Scholes closed-form solution and report
the variance reduction achieved.

Author: Petre Maniu
=============================================================
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Black-Scholes closed-form solutions
# ---------------------------------------------------------------------------

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    We compute the Black-Scholes price of a European call option.

    Under the risk-neutral measure Q, the price is given by the discounted
    expectation of the terminal payoff:

        C = e^{-rT} * E^Q[ max(S_T - K, 0) ]

    where S_T ~ log-normal. The closed-form solution, obtained via the
    Feynman-Kac representation of the Black-Scholes PDE, is:

        C = S * N(d1) - K * e^{-rT} * N(d2)

    with d1 = [ ln(S/K) + (r + sigma^2/2) * T ] / (sigma * sqrt(T))
    and  d2 = d1 - sigma * sqrt(T),
    where N denotes the standard normal CDF.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    We compute the Black-Scholes price of a European put option.

    Rather than re-deriving the put price independently, we appeal to
    put-call parity, which holds by a no-arbitrage argument:

        P = C - S + K * e^{-rT}

    This identity follows from the fact that a long call and short put
    with the same strike and expiry replicates a forward contract.
    """
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# 2. GBM path simulation via Euler-Maruyama discretisation
# ---------------------------------------------------------------------------

def simulate_gbm(
    S0: float, r: float, sigma: float, T: float,
    n_steps: int, n_paths: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    We simulate sample paths of Geometric Brownian Motion under the
    risk-neutral measure by discretising the log-price SDE via
    Euler-Maruyama.

    The continuous dynamics are given by:
        dS_t = r * S_t dt + sigma * S_t dW_t

    Applying Ito's lemma to ln(S_t) yields the exact increment:
        ln(S_{t+dt}) = ln(S_t) + (r - sigma^2/2) * dt + sigma * sqrt(dt) * Z

    where Z ~ N(0,1). We note that, unlike discretising S_t directly,
    this update introduces no discretisation error in the log-price:
    the paths are exact realisations of GBM at the chosen time grid.

    We return the shock matrix Z alongside the paths so that the antithetic
    counterpart (-Z) may be constructed without a further RNG call, which
    is essential for the variance reduction scheme described below.

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated price paths. Column 0 holds S0; columns 1..n_steps
        hold S_{dt}, S_{2dt}, ..., S_T.
    Z : ndarray, shape (n_paths, n_steps)
        The i.i.d. N(0,1) shocks from which the paths are constructed.
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    Z              = rng.standard_normal((n_paths, n_steps))           # (P, M)
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths      = np.log(S0) + np.cumsum(log_increments, axis=1)   # (P, M)

    paths = np.concatenate(
        [np.full((n_paths, 1), S0), np.exp(log_paths)], axis=1        # (P, M+1)
    )
    return paths, Z


def _build_paths(S0: float, r: float, sigma: float, T: float,
                 n_steps: int, Z: np.ndarray) -> np.ndarray:
    """
    We reconstruct GBM paths from a pre-existing shock matrix Z.

    This auxiliary function exists solely to support the antithetic
    variates construction: passing -Z in place of Z yields the
    antithetic path, which is correlated with the original by design.
    We avoid a second call to the RNG, which would produce independent
    rather than antithetic samples.
    """
    dt             = T / n_steps
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths      = np.log(S0) + np.cumsum(log_increments, axis=1)
    return np.concatenate(
        [np.full((Z.shape[0], 1), S0), np.exp(log_paths)], axis=1
    )


# ---------------------------------------------------------------------------
# 3. Variance reduction via antithetic variates
# ---------------------------------------------------------------------------

def _price_and_se(
    payoffs: np.ndarray, payoffs_anti: np.ndarray,
    r: float, T: float
) -> tuple[float, float]:
    """
    We combine a pair of antithetic payoff arrays into a single discounted
    price estimate together with its Monte Carlo standard error.

    The antithetic variates method rests on the following observation.
    Let f be a monotone function (as all option payoffs are). Then:

        Cov( f(Z), f(-Z) ) <= 0

    It follows that:

        Var( 0.5 * (f(Z) + f(-Z)) ) = 0.5 * Var(f(Z)) + 0.5 * Cov(f(Z), f(-Z))
                                     <= 0.5 * Var(f(Z))

    so the paired estimator is at least as efficient as the naive estimator
    at half the path count, with strict improvement whenever the covariance
    is strictly negative.

    We note that the standard error must be computed on the paired averages
    — one observation per path-pair — and not on the raw payoffs. The latter
    would overstate the SE by treating the two antithetic samples as
    independent, thereby ignoring the variance reduction already achieved.

    Returns
    -------
    price : float   Discounted expectation: e^{-rT} * E[payoff]
    se    : float   Standard error of the price estimate
    """
    paired   = 0.5 * (payoffs + payoffs_anti)     # one observation per path-pair
    discount = np.exp(-r * T)
    price    = discount * paired.mean()
    se       = discount * paired.std(ddof=1) / np.sqrt(len(paired))
    return price, se


def _naive_se(payoffs: np.ndarray, r: float, T: float) -> float:
    """
    We compute the standard error of the naive Monte Carlo estimator,
    i.e. without any variance reduction. This serves as a baseline against
    which the efficiency gain from antithetic variates is measured.
    """
    discount = np.exp(-r * T)
    return discount * payoffs.std(ddof=1) / np.sqrt(len(payoffs))


# ---------------------------------------------------------------------------
# 4. Option pricers
# ---------------------------------------------------------------------------

def price_european(
    S0: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
    n_steps: int = 252,
    n_paths: int = 100_000,
    seed: int = 42
) -> tuple[float, float]:
    """
    We price a European option by Monte Carlo simulation with antithetic
    variates. The payoff is a function of the terminal price S_T alone:

        Call payoff: max(S_T - K, 0)
        Put  payoff: max(K - S_T, 0)

    We draw paths under the risk-neutral measure, evaluate the discounted
    expected payoff, and return the price together with its standard error.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Initial spot price, strike, time to expiry (in years), continuously
        compounded risk-free rate, and volatility.
    option_type : {"call", "put"}
    n_steps     : int   Number of time steps; 252 corresponds to daily steps
                        over a one-year horizon.
    n_paths     : int   Number of simulated paths; larger values yield
                        lower standard errors at O(1/sqrt(n)) rate.
    seed        : int   Seed for the pseudo-random number generator,
                        ensuring reproducibility.

    Returns
    -------
    price : float
    se    : float   Monte Carlo standard error of the price estimate.
    """
    _, Z = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed)

    # We extract only the terminal values S_T; intermediate prices are
    # irrelevant for European payoffs and need not be retained.
    ST      = _build_paths(S0, r, sigma, T, n_steps,  Z)[:, -1]
    ST_anti = _build_paths(S0, r, sigma, T, n_steps, -Z)[:, -1]

    if option_type == "call":
        pay, pay_anti = np.maximum(ST - K, 0), np.maximum(ST_anti - K, 0)
    else:
        pay, pay_anti = np.maximum(K - ST, 0), np.maximum(K - ST_anti, 0)

    return _price_and_se(pay, pay_anti, r, T)


def price_asian_arithmetic(
    S0: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
    n_steps: int = 252,
    n_paths: int = 100_000,
    seed: int = 42
) -> tuple[float, float]:
    """
    We price a fixed-strike arithmetic Asian option by Monte Carlo with
    antithetic variates. The payoff depends on the arithmetic mean of S
    over the n_steps monitoring dates {dt, 2dt, ..., T}:

        Call payoff: max( (1/M) * sum_{k=1}^{M} S_{k*dt} - K, 0 )
        Put  payoff: max( K - (1/M) * sum_{k=1}^{M} S_{k*dt}, 0 )

    We note that S_0 is excluded from the average, as the initial price is
    known and deterministic; including it would introduce a downward bias
    in the mean and is inconsistent with the standard fixed-strike Asian
    convention.

    Asian options are strictly cheaper than their European counterparts
    (for the same K and T) because the arithmetic mean has lower variance
    than the terminal price, which can be seen from the sub-additivity of
    the variance functional applied to the partial sums.

    Returns
    -------
    price : float
    se    : float
    """
    _, Z = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed)

    # We index [:, 1:] to restrict the mean to the M monitoring dates,
    # deliberately excluding the deterministic initial value S_0.
    avg      = _build_paths(S0, r, sigma, T, n_steps,  Z)[:, 1:].mean(axis=1)
    avg_anti = _build_paths(S0, r, sigma, T, n_steps, -Z)[:, 1:].mean(axis=1)

    if option_type == "call":
        pay, pay_anti = np.maximum(avg - K, 0), np.maximum(avg_anti - K, 0)
    else:
        pay, pay_anti = np.maximum(K - avg, 0), np.maximum(K - avg_anti, 0)

    return _price_and_se(pay, pay_anti, r, T)


# ---------------------------------------------------------------------------
# 5. Benchmark and variance reduction report
# ---------------------------------------------------------------------------

def run_benchmark(
    S0: float = 100.0, K: float = 100.0, T: float = 1.0,
    r: float = 0.05, sigma: float = 0.20,
    n_paths: int = 200_000, n_steps: int = 252
) -> None:
    """
    We benchmark the Monte Carlo prices against the Black-Scholes
    analytical solution and quantify the variance reduction achieved
    by the antithetic variates method.

    We expect the following:
    - European MC prices lie within approximately one to two standard
      errors of the corresponding Black-Scholes value, confirming that
      our estimator is unbiased.
    - Asian prices are strictly below their European counterparts, which
      is consistent with the lower variance of the time-averaged payoff.
    - The antithetic standard error is materially smaller than the naive
      standard error, with the ratio of squared standard errors giving
      the effective path-count gain.
    """
    print("=" * 65)
    print("Monte Carlo Option Pricing — Black-Scholes Benchmark")
    print("=" * 65)
    print(f"  S={S0}  K={K}  T={T}yr  r={r:.0%}  σ={sigma:.0%}")
    print(f"  Paths: {n_paths:,}   Steps per path: {n_steps}\n")

    header = f"{'':6} {'BS Analytical':>14} {'MC European':>22} {'MC Asian':>22}"
    print(header)
    print("-" * len(header))

    for opt in ("call", "put"):
        bs = bs_call(S0, K, T, r, sigma) if opt == "call" else bs_put(S0, K, T, r, sigma)
        mc_eur, se_eur = price_european(S0, K, T, r, sigma, opt, n_steps, n_paths)
        mc_asi, se_asi = price_asian_arithmetic(S0, K, T, r, sigma, opt, n_steps, n_paths)

        err_eur = mc_eur - bs
        print(
            f"  {opt.upper():<5}"
            f"  {bs:>12.4f}"
            f"    {mc_eur:>8.4f} ±{se_eur:.4f}  (err {err_eur:+.4f})"
            f"    {mc_asi:>8.4f} ±{se_asi:.4f}"
        )

    # We isolate the European call to illustrate the variance reduction
    # in concrete numerical terms.
    print("\n" + "=" * 65)
    print("Variance Reduction: Antithetic vs Naive MC  (European call)")
    print("=" * 65)

    _, Z    = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed=42)
    ST      = _build_paths(S0, r, sigma, T, n_steps,  Z)[:, -1]
    ST_anti = _build_paths(S0, r, sigma, T, n_steps, -Z)[:, -1]

    pay_naive     = np.maximum(ST - K, 0)
    pay_anti      = np.maximum(ST_anti - K, 0)
    se_naive      = _naive_se(pay_naive, r, T)
    _, se_antith  = _price_and_se(pay_naive, pay_anti, r, T)
    reduction_pct = 100 * (1 - se_antith / se_naive)
    path_gain     = (se_naive / se_antith) ** 2

    print(f"  Naive MC SE          : {se_naive:.6f}")
    print(f"  Antithetic MC SE     : {se_antith:.6f}")
    print(f"  Variance reduction   : {reduction_pct:.1f}%")
    print(f"  Equivalent path gain : {path_gain:.1f}x  "
          f"(antithetic precision matches ~{path_gain:.0f}x more naive paths)")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()