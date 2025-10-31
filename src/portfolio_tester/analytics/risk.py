import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_cov_annual(returns_m: pd.DataFrame):
    mu_m = returns_m.mean()
    cov_m = returns_m.cov()
    mu_a = (1 + mu_m)**12 - 1
    cov_a = cov_m * 12
    return mu_a.values, cov_a.values, list(returns_m.columns)

def _solve_minvar(mu, cov, target_ret, lb=0.0, ub=1.0):
    n = len(mu)
    bounds = [(lb, ub)] * n
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target_ret},
    )
    x0 = np.ones(n)/n
    res = minimize(lambda w: w @ cov @ w, x0, bounds=bounds, constraints=cons)
    if not res.success:
        return None
    return res.x

def efficient_frontier(returns_m: pd.DataFrame, n_pts: int = 50):
    mu, cov, names = mean_cov_annual(returns_m)
    ret_min, ret_max = mu.min(), mu.max()
    targets = np.linspace(ret_min, ret_max, n_pts)
    weights, risks, rets = [], [], []
    for tr in targets:
        w = _solve_minvar(mu, cov, tr)
        if w is not None:
            weights.append(w)
            rets.append(tr)
            risks.append(np.sqrt(w @ cov @ w))
    return np.array(weights), np.array(rets), np.array(risks), names

def portfolio_annual_stats(weights, returns_m: pd.DataFrame):
    mu_m = (returns_m * weights).sum(axis=1).mean()
    vol_m = (returns_m @ weights).std(ddof=1)
    mu_a = (1 + mu_m)**12 - 1
    vol_a = vol_m * np.sqrt(12)
    return mu_a, vol_a

def corr_table(returns_m: pd.DataFrame):
    return returns_m.corr()
