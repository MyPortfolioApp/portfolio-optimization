import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_cov_annual(returns_m: pd.DataFrame):
    mu_m = returns_m.mean()
    cov_m = returns_m.cov()
    mu_a = (1 + mu_m)**12 - 1
    cov_a = cov_m * 12
    return mu_a.values, cov_a.values, list(returns_m.columns)

def risk_free_annual(rf_m):
    """
    Annualize a monthly risk-free series (geometric).
    rf_m: pandas.Series of monthly rates (e.g., TB3MS converted to monthly).
    """
    import numpy as np
    rf_m = np.asarray(rf_m.dropna(), dtype=float)
    if rf_m.size == 0:
        return 0.0
    rf_a = (1.0 + rf_m).prod() ** (12.0 / rf_m.size) - 1.0
    return float(rf_a)

def _solve_gmv(cov, lb=0.0, ub=1.0):
    """Long-only GMV portfolio: min variance s.t. sum(weights)=1."""
    import numpy as np
    from scipy.optimize import minimize

    n = cov.shape[0]
    bounds = [(lb, ub)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w, x0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

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

def _solve_max_sharpe(mu, cov, rf_a, lb=0.0, ub=1.0):
    """
    Long-only max Sharpe: maximize (w·(mu - rf)) / sqrt(w·Cov·w)
    s.t. sum(w)=1, 0<=w<=1.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = len(mu)
    bounds = [(lb, ub)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        ex = mu - rf_a
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        ret = w @ mu
        return - (ret - rf_a) / vol

    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

def max_sharpe_portfolio(returns_m, rf_m):
    """
    Returns (weights, ret_a, vol_a, sharpe, rf_a, names)
    using long-only constraints and sum(w)=1.
    """
    import numpy as np
    mu, cov, names = mean_cov_annual(returns_m)
    rf_a = risk_free_annual(rf_m)
    w = _solve_max_sharpe(mu, cov, rf_a)
    if w is None:
        return None, np.nan, np.nan, np.nan, rf_a, names
    ret_a = float(w @ mu)
    vol_a = float(np.sqrt(w @ cov @ w))
    sharpe = (ret_a - rf_a) / max(vol_a, 1e-16)
    return w, ret_a, vol_a, sharpe, rf_a, names

def efficient_frontier(returns_m, n_pts: int = 50):
    # annualized mean & covariance
    mu, cov, names = mean_cov_annual(returns_m)
    # GMV portfolio return — this splits lower/upper branches
    w_gmv = _solve_gmv(cov)
    r_gmv = float(w_gmv @ mu) if w_gmv is not None else float(mu.min())
    # Only target returns on/above GMV return → upper (efficient) branch
    targets = np.linspace(r_gmv, float(mu.max()), n_pts)

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

def single_asset_stats(returns_m):
    """Annualized (vol, return) per asset for scatter labels."""
    import numpy as np
    mu_m = returns_m.mean()
    mu_a = (1 + mu_m)**12 - 1
    vol_m = returns_m.std(ddof=1)
    vol_a = vol_m * np.sqrt(12)
    names = list(returns_m.columns)
    return vol_a.values, mu_a.values, names


def corr_table(returns_m: pd.DataFrame):
    return returns_m.corr()
