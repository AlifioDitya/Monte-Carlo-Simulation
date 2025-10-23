
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple
from math import erf, sqrt, log
import numpy as np
from datetime import datetime

# ---------------------- Data classes ----------------------

@dataclass
class GoalParams:
    W0: float
    target_FV: float
    horizon_months: int
    contributions: float | Sequence[float]  # start-of-month; scalar or array length n

@dataclass
class MarketParams:
    # Either provide monthly lognormal params...
    m_month: Optional[float] = None
    s_month: Optional[float] = None
    # ...or annual arithmetic assumptions (converted internally)
    annual_return: Optional[float] = None
    annual_vol: Optional[float] = None

@dataclass
class SimulationConfig:
    paths: int = 2000
    seed: Optional[int] = 12345
    store_paths: bool = True             # store full path matrix (needed for projection band)
    max_paths_for_storage: int = 5000    # cap for memory control

# ---------------------- Conversions & helpers ----------------------

def _to_monthly_params(market: MarketParams) -> Tuple[float, float, float]:
    """Return (m, s, a) where:
       m = monthly log-mean; s = monthly log-std; a = E[G]_month (expected monthly gross).
       If monthly params are given, use them; else convert from annual arithmetic.
    """
    if market.m_month is not None and market.s_month is not None:
        m = float(market.m_month)
        s = float(market.s_month)
        a = float(np.exp(m + 0.5 * s * s))
        return m, s, a
    if market.annual_return is None or market.annual_vol is None:
        raise ValueError("Provide either (m_month, s_month) or (annual_return, annual_vol).")
    E_G_month = float((1.0 + float(market.annual_return)) ** (1.0 / 12.0))
    s_month = float(market.annual_vol) / float(np.sqrt(12.0))
    m_month = float(np.log(E_G_month) - 0.5 * (s_month ** 2))
    return m_month, s_month, E_G_month

def _normalize_contributions(contribs: float | Sequence[float], n: int) -> np.ndarray:
    if isinstance(contribs, (int, float)):
        return np.full(n, float(contribs), dtype=float)
    arr = np.asarray(contribs, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(f"contributions length must be {n}, got {arr.size}")
    return arr

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# ---------------------- Core simulation ----------------------

def simulate_goal_mc(goal: GoalParams, market: MarketParams, cfg: SimulationConfig) -> Dict[str, Any]:
    """Monte Carlo goal simulation with start-of-month contributions.
    State update: W_t = (W_{t-1} + c_t) * G_t, with G_t = exp(m + s * Z_t), Z ~ N(0,1).
    Returns:
        dict with:
          - as_of (ISO timestamp)
          - inputs/as_of_params
          - end_values (paths,)
          - prob_completion (float)
          - percentiles {p10, p50, p90}
          - time_grid (0..n)
          - projection_band {q10, q50, q90} across time
          - paths_sample (n+1, k) small subset for plotting
    """
    m, s, E_G_month = _to_monthly_params(market)
    n = int(goal.horizon_months)
    contribs = _normalize_contributions(goal.contributions, n)
    rng = np.random.default_rng(cfg.seed)

    paths = int(cfg.paths)
    store_paths = bool(cfg.store_paths) and (paths <= cfg.max_paths_for_storage)
    if store_paths:
        W = np.zeros((n + 1, paths), dtype=float)
        W[0, :] = goal.W0
    else:
        # if not storing, keep rolling end values, and separate light sample for plotting
        W = None

    # For sample visualization always keep up to 200 paths
    sample_k = min(paths, 200)
    W_sample = np.zeros((n + 1, sample_k), dtype=float)
    W_sample[0, :] = goal.W0

    # Simulate
    if store_paths:
        for t in range(1, n + 1):
            Z = rng.standard_normal(paths)
            G = np.exp(m + s * Z)
            W[t, :] = (W[t - 1, :] + contribs[t - 1]) * G
            # update sample
            W_sample[t, :] = W[t, :sample_k]
        end_vals = W[-1, :].copy()
    else:
        # No full storage: we still need projection band; so in practice we advise enabling storage for small n.
        # Here we simulate and stack into chunks (still fine for typical sizes).
        W_accum = []
        for start in range(0, paths, 1000):
            take = min(1000, paths - start)
            W_chunk = np.zeros((n + 1, take), dtype=float)
            W_chunk[0, :] = goal.W0
            for t in range(1, n + 1):
                Z = rng.standard_normal(take)
                G = np.exp(m + s * Z)
                W_chunk[t, :] = (W_chunk[t - 1, :] + contribs[t - 1]) * G
                if start == 0 and t <= n and take >= sample_k:
                    W_sample[t, :] = W_chunk[t, :sample_k]
            W_accum.append(W_chunk)
        W = np.concatenate(W_accum, axis=1)
        end_vals = W[-1, :]

    # Probability and end distribution
    prob = float(np.mean(end_vals >= goal.target_FV))
    p10, p50, p90 = np.percentile(end_vals, [10, 50, 90])

    # Projection band across time
    q10 = np.percentile(W, 10, axis=1)
    q50 = np.percentile(W, 50, axis=1)
    q90 = np.percentile(W, 90, axis=1)

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "as_of_params": {
            "W0": float(goal.W0),
            "target_FV": float(goal.target_FV),
            "horizon_months": n,
            "contributions_sum": float(contribs.sum()),
            "m_month": float(m),
            "s_month": float(s),
            "E_G_month": float(E_G_month),
            "paths": paths,
            "seed": cfg.seed,
        },
        "end_values": end_vals,
        "prob_completion": prob,
        "percentiles": {"p10": float(p10), "p50": float(p50), "p90": float(p90)},
        "time_grid": np.arange(n + 1, dtype=int),
        "projection_band": {"q10": q10, "q50": q50, "q90": q90},
        "paths_sample": W_sample,
    }

# ---------------------- Analytic & deterministic helpers ----------------------

def fenton_wilkinson_probability(goal: GoalParams, market: MarketParams) -> Dict[str, float]:
    """Analytic probability via Fenton–Wilkinson moment matching on end wealth W_n.
       Returns {'mu1': mean, 'var': variance, 'p': probability}
    """
    m, s, E_G = _to_monthly_params(market)
    a = float(np.exp(m + 0.5 * s * s))
    b = float(np.exp(2.0 * m + 2.0 * s * s))

    n = int(goal.horizon_months)
    contribs = _normalize_contributions(goal.contributions, n)

    # Mean
    mu1 = float(goal.W0 * (a ** n) + sum(contribs[t] * (a ** (n - t)) for t in range(n)) * a)

    # Second moment
    # E[Z_i Z_j] = b^{n-max(i,j)+1} * a^{|i-j|}
    mu2 = float((goal.W0 ** 2) * (b ** n))
    # cross term with W0
    cross = 0.0
    for k in range(1, n + 1):
        cross += (b ** n) * (a ** (k - 1)) * contribs[k - 1]
    mu2 += 2.0 * goal.W0 * cross
    # sum of contrib terms
    ssum = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            ssum += contribs[i - 1] * contribs[j - 1] * (b ** (n - max(i, j) + 1)) * (a ** abs(i - j))
    mu2 += ssum

    var = max(mu2 - mu1 * mu1, 0.0)
    if mu1 <= 0.0:
        return {"mu1": mu1, "var": var, "p": 0.0}
    tilde_sigma2 = float(np.log(1.0 + var / (mu1 * mu1))) if var > 0 else 0.0
    tilde_sigma = float(np.sqrt(max(tilde_sigma2, 0.0)))
    tilde_mu = float(np.log(mu1) - 0.5 * tilde_sigma2) if tilde_sigma2 > 0 else float(np.log(mu1))
    if tilde_sigma > 0:
        z = (np.log(goal.target_FV) - tilde_mu) / tilde_sigma
        p = 1.0 - _norm_cdf(z)
    else:
        p = 1.0 if mu1 >= goal.target_FV else 0.0
    return {"mu1": mu1, "var": var, "p": p}

def deterministic_end_wealth(goal: GoalParams, market: MarketParams) -> float:
    """Deterministic end wealth using expected monthly gross a (no volatility)."""
    m, s, E_G = _to_monthly_params(market)
    a = float(np.exp(m + 0.5 * s * s))
    n = int(goal.horizon_months)
    contribs = _normalize_contributions(goal.contributions, n)
    Wn = goal.W0 * (a ** n)
    for t in range(n):
        Wn += contribs[t] * (a ** (n - t)) * a  # start-of-month contribution
    return float(Wn)

# ---------------------- Plot helpers (matplotlib) ----------------------

def plot_projection_band(time_grid: np.ndarray, band: Dict[str, np.ndarray], target_FV: float) -> None:
    """Single-figure P10–P90 band with median line and target at horizon."""
    import matplotlib.pyplot as plt
    q10, q50, q90 = band["q10"], band["q50"], band["q90"]
    plt.figure()
    plt.plot(time_grid, q50, label="Median")
    plt.fill_between(time_grid, q10, q90, alpha=0.3, label="P10–P90 band")
    plt.axhline(y=target_FV, linestyle="--", label="Target at horizon")
    plt.title("Projection Band (Monthly)")
    plt.xlabel("Month")
    plt.ylabel("Wealth")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_simulated_paths(time_grid: np.ndarray, paths_sample: np.ndarray, target_FV: float) -> None:
    """Plot a subset of simulated wealth paths and the target line."""
    import matplotlib.pyplot as plt
    plt.figure()
    for k in range(paths_sample.shape[1]):
        plt.plot(time_grid, paths_sample[:, k])
    plt.axhline(y=target_FV, linestyle="--", label="Target")
    plt.title(f"Simulated Wealth Paths ({paths_sample.shape[1]})")
    plt.xlabel("Month")
    plt.ylabel("Wealth")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_end_wealth_distribution(end_values: np.ndarray, target_FV: float, prob_completion: float) -> None:
    """Histogram of end wealth with target line and probability in title."""
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(end_values, bins=40)
    plt.axvline(target_FV, linestyle="--", label="Target")
    plt.title(f"End Wealth Distribution (n={end_values.size})\nP(W_n ≥ Target) = {prob_completion*100:.1f}%")
    plt.xlabel("End Wealth")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
