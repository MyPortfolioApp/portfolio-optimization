from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy.optimize import minimize

from portfolio_tester.analytics.risk import mean_cov_annual
from portfolio_tester.config import Goal, SamplerConfig, SimConfig
from portfolio_tester.engine.simulator import MonteCarloSimulator
from portfolio_tester.sampling.bootstrap import ReturnSampler


@dataclass
class OptimizationConstraints:
    """
    Describes per-asset weight bounds and optional volatility caps used when
    building a constrained mean-variance frontier.
    """

    min_weights: Sequence[float] | float | None = None
    max_weights: Sequence[float] | float | None = None
    allow_short: bool = False
    max_volatility: float | None = None  # annualized volatility cap

    def expand_bounds(self, n_assets: int) -> List[tuple[float, float]]:
        """
        Convert scalar/sequence bounds into a list of (lb, ub) tuples suitable
        for scipy's SLSQP optimizer.
        """
        def _coerce(values, default):
            if values is None:
                return np.full(n_assets, default, dtype=float)
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 0:
                return np.full(n_assets, float(arr), dtype=float)
            if arr.size != n_assets:
                raise ValueError(
                    f"Constraint length mismatch: expected {n_assets}, got {arr.size}"
                )
            return arr.astype(float)

        default_min = -1.0 if self.allow_short else 0.0
        default_max = 1.0
        lb = _coerce(self.min_weights, default_min)
        ub = _coerce(self.max_weights, default_max)
        if not self.allow_short:
            lb = np.maximum(lb, 0.0)
        if np.any(lb > ub):
            raise ValueError("Invalid constraints: min_weights exceed max_weights.")
        return list(zip(lb.tolist(), ub.tolist()))


@dataclass
class FrontierPortfolio:
    weights: np.ndarray
    expected_return: float
    volatility: float
    survival_rate: float | None = None
    smoothed_survival: float | None = None


@dataclass
class SurvivalOptimizationResult:
    best_portfolio: FrontierPortfolio
    frontier: list[FrontierPortfolio]
    best_index: int
    smoothed_survival_rates: np.ndarray
    best_simulation_output: dict | None = field(default=None, repr=False)


def _initial_weights(bounds: List[tuple[float, float]]) -> np.ndarray:
    """
    Build a feasible starting point by taking the midpoint of each bound. The
    optimizer (SLSQP) will enforce the equality constraints, so this only needs
    to respect the box constraints.
    """
    mids = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)
    # If all mids sum to ~0 (possible w/ symmetric short bounds), fall back
    if np.allclose(mids.sum(), 0.0):
        mids = np.array([hi for _, hi in bounds], dtype=float)
    return mids


def _solve_portfolio(
    cov: np.ndarray,
    bounds: List[tuple[float, float]],
    mu: np.ndarray | None = None,
    target_return: float | None = None,
) -> np.ndarray | None:
    """
    Solve: minimize w^T Cov w
           subject to sum(w)=1 and optionally w·mu = target_return.
    """
    n = cov.shape[0]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if mu is not None and target_return is not None:
        cons.append({"type": "eq", "fun": lambda w, tr=target_return: w @ mu - tr})
    x0 = _initial_weights(bounds)
    res = minimize(lambda w: w @ cov @ w, x0, bounds=bounds, constraints=cons)
    if not res.success:
        return None
    return res.x


def build_constrained_frontier(
    returns_m,
    constraints: OptimizationConstraints,
    n_portfolios: int = 50,
) -> list[FrontierPortfolio]:
    """
    Construct a constrained efficient frontier that honors per-asset bounds and
    optional volatility caps. Returns a list of FrontierPortfolio entries.
    """
    if n_portfolios < 2:
        raise ValueError("n_portfolios must be >= 2.")

    mu, cov, _ = mean_cov_annual(returns_m)
    n_assets = len(mu)
    bounds = constraints.expand_bounds(n_assets)

    w_gmv = _solve_portfolio(cov, bounds)
    if w_gmv is None:
        raise RuntimeError("Failed to solve for the global minimum-variance portfolio.")

    ret_min = float(w_gmv @ mu)
    ret_max = float(np.max(mu))
    if np.isclose(ret_min, ret_max):
        targets = np.array([ret_min] * n_portfolios, dtype=float)
    else:
        targets = np.linspace(ret_min, ret_max, n_portfolios)

    frontier: list[FrontierPortfolio] = []
    for target in targets:
        w = _solve_portfolio(cov, bounds, mu=mu, target_return=target)
        if w is None:
            continue
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        if constraints.max_volatility is not None and vol > constraints.max_volatility:
            continue
        frontier.append(
            FrontierPortfolio(weights=w.astype(float), expected_return=ret, volatility=vol)
        )

    if not frontier:
        raise RuntimeError("No feasible portfolios were found under the provided constraints.")
    return frontier


def evaluate_survival_for_weights(
    weights: np.ndarray,
    sim_config: SimConfig,
    sampler: ReturnSampler,
    goals: Sequence[Goal],
    sampler_config: SamplerConfig,
    n_sims: int | None = None,
    return_full_output: bool = False,
) -> tuple[float, dict | None]:
    """
    Run the Monte Carlo engine for the provided weights and return the survival
    rate (share of paths with positive ending balance). Optionally return the
    raw simulator output.
    """
    sims = int(n_sims) if n_sims is not None else sim_config.n_sims
    R_paths, CPI_paths = sampler.sample(sim_config.horizon_months, sims, sampler_config)
    simulator = MonteCarloSimulator(
        weights=weights,
        starting_balance=sim_config.starting_balance,
        rebalance_every_months=sim_config.rebalance_every_months,
    )
    out = simulator.run_with_cashflows(R_paths, CPI_paths, goals)
    out["infl_paths"] = CPI_paths
    survival_rate = float((out["balances"][:, -1] > 0).mean())
    if return_full_output:
        return survival_rate, out
    return survival_rate, None


def compute_survival_on_frontier(
    frontier: list[FrontierPortfolio],
    sim_config: SimConfig,
    sampler: ReturnSampler,
    goals: Sequence[Goal],
    sampler_config: SamplerConfig,
    n_sims_per_portfolio: int = 1000,
) -> None:
    """
    Mutates the frontier entries by populating their survival rates using the
    Monte Carlo simulator with a moderate number of simulations.
    """
    for fp in frontier:
        survival, _ = evaluate_survival_for_weights(
            fp.weights,
            sim_config=sim_config,
            sampler=sampler,
            goals=goals,
            sampler_config=sampler_config,
            n_sims=n_sims_per_portfolio,
            return_full_output=False,
        )
        fp.survival_rate = survival


def smooth_survival_rates(survival_rates: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply a simple moving average smoothing with the requested window size.
    The window is centered on each point; edge points shrink the window to the
    available neighbors.
    """
    if window <= 1:
        return survival_rates.copy()
    smoothed = np.zeros_like(survival_rates, dtype=float)
    half = window // 2
    n = survival_rates.size
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed[i] = survival_rates[start:end].mean()
    return smoothed


def find_best_survival_portfolio(
    frontier: list[FrontierPortfolio],
    smoothing_window: int = 3,
) -> SurvivalOptimizationResult:
    """
    Sort the frontier by volatility, smooth the survival series, and identify
    the index with the highest smoothed survival rate (breaking ties by raw
    survival, then expected return).
    """
    if not frontier:
        raise ValueError("Frontier is empty.")
    sorted_frontier = sorted(frontier, key=lambda fp: fp.volatility)
    survival = np.array(
        [fp.survival_rate for fp in sorted_frontier], dtype=float
    )
    if np.any(np.isnan(survival)):
        raise ValueError("Survival rates must be computed before optimization.")
    smoothed = smooth_survival_rates(survival, window=max(1, smoothing_window))
    max_val = smoothed.max()
    candidates = np.where(np.isclose(smoothed, max_val))[0]

    def _tie_break(idx: int) -> tuple[float, float]:
        return survival[idx], sorted_frontier[idx].expected_return

    best_idx = int(max(candidates, key=_tie_break))
    for fp, smooth_val in zip(sorted_frontier, smoothed):
        fp.smoothed_survival = float(smooth_val)
    return SurvivalOptimizationResult(
        best_portfolio=sorted_frontier[best_idx],
        frontier=sorted_frontier,
        best_index=best_idx,
        smoothed_survival_rates=smoothed,
    )


def refine_best_portfolio(
    result: SurvivalOptimizationResult,
    sim_config: SimConfig,
    sampler: ReturnSampler,
    goals: Sequence[Goal],
    sampler_config: SamplerConfig,
    n_sims_best: int = 10_000,
) -> None:
    """
    Re-run the Monte Carlo engine for the best portfolio with a larger sample
    size to obtain more stable survival metrics and retain the raw output for
    downstream reporting/plotting.
    """
    survival, out = evaluate_survival_for_weights(
        result.best_portfolio.weights,
        sim_config=sim_config,
        sampler=sampler,
        goals=goals,
        sampler_config=sampler_config,
        n_sims=n_sims_best,
        return_full_output=True,
    )
    result.best_portfolio.survival_rate = survival
    result.best_simulation_output = out
