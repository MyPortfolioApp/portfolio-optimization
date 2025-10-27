import numpy as np

def cagr(balances, months: int):
    end = balances[:, -1]
    start = balances[:, 0]
    years = months / 12.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(start > 0, (end / start) ** (1/years) - 1.0, np.nan)

def twrr_annualized(twrr_monthly):
    g = np.prod(1.0 + twrr_monthly, axis=1)
    years = twrr_monthly.shape[1] / 12.0
    return g ** (1/years) - 1.0

def mwrr_irr(cashflows, balances):
    import numpy_financial as npf
    n_sims, T = cashflows.shape
    irr = np.full(n_sims, np.nan)
    for s in range(n_sims):
        series = [-balances[s,0]] + cashflows[s].tolist() + [balances[s,-1]]
        try:
            irr[s] = npf.irr(series) * 12.0
        except Exception:
            irr[s] = np.nan
    return irr

def sharpe_sortino(twrr_monthly, rf_m):
    ex = twrr_monthly - rf_m.values[None, :twrr_monthly.shape[1]]
    mean_m = ex.mean(axis=1)
    vol_m = ex.std(axis=1, ddof=1)
    downside = np.where(ex < 0, ex, 0.0)
    dvol_m = downside.std(axis=1, ddof=1)
    sharpe = (mean_m / np.where(vol_m>0, vol_m, np.nan)) * np.sqrt(12)
    sortino = (mean_m / np.where(dvol_m>0, dvol_m, np.nan)) * np.sqrt(12)
    return sharpe, sortino

def max_drawdown(paths):
    mdds = np.zeros(paths.shape[0])
    for s in range(paths.shape[0]):
        x = paths[s]
        peak = np.maximum.accumulate(x)
        dd = (x - peak) / peak
        mdds[s] = dd.min()
    return mdds
