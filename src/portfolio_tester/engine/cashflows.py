import numpy as np
from ..config import Goal

def _step_months_from_frequency(freq: int) -> int:
    # freq is payments per year: 12->1 month, 4->3 months, 1->12 months
    return int(12 // freq)

def build_cashflow_vector(goals, horizon_m: int, infl_path=None):
    """Return a (horizon_m,) vector of end-of-month cashflows.
    Positive = contribution (deposit), Negative = withdrawal.
    If goal.real is True, amounts are indexed by cumulative (1+infl) to the payment date.
    """
    cf = np.zeros(horizon_m, dtype=float)
    infl_cum = None
    if infl_path is not None:
        infl_cum = np.cumprod(1.0 + infl_path)
    for g in goals:
        due = int(g.start_month)
        step = _step_months_from_frequency(int(g.frequency))
        for _ in range(int(g.repeats)):
            if due < horizon_m:
                amt = float(g.amount)
                if g.real and infl_cum is not None and due > 0:
                    amt *= infl_cum[due-1]
                cf[due] += amt
            due += step
    return cf


def _monthly_to_annual_sum(arr_m: np.ndarray, horizon_months: int) -> np.ndarray:
    """
    Sum monthly series into years.
    arr_m: shape (n_sims, T)
    returns: shape (n_sims, years)
    """
    n_sims, T = arr_m.shape
    years = int(np.ceil(horizon_months / 12))
    out = np.zeros((n_sims, years), dtype=float)
    for y in range(years):
        start = y * 12
        end = min((y + 1) * 12, T)
        if start >= T:
            break
        out[:, y] = arr_m[:, start:end].sum(axis=1)
    return out

def _deflate_to_present_dollars(cf_m: np.ndarray, infl_paths: np.ndarray) -> np.ndarray:
    """
    Convert nominal monthly cashflows to present dollars (t=0) per path.
    Uses the same convention as build_cashflow_vector: real payments were
    indexed by cumulative CPI up to the previous month (infl_cum[due-1]).
    """
    n_sims, T = cf_m.shape
    real = np.zeros_like(cf_m)
    for s in range(n_sims):
        infl = infl_paths[s]                 # shape (T,)
        infl_cum = np.cumprod(1.0 + infl)    # [t] = cumulative through month t
        real[s, 0] = cf_m[s, 0]
        if T > 1:
            # month t >= 1 → divide by inflation up to t-1
            real[s, 1:] = cf_m[s, 1:] / infl_cum[:-1]
    return real

def annual_cashflow_medians(cf_m: np.ndarray, infl_paths: np.ndarray, horizon_months: int):
    """
    From monthly cashflows + inflation paths, return:
      - nominal_median_by_year: median of annual nominal cashflows
      - real_median_by_year:    median of annual present-dollar cashflows
    Shapes:
      cf_m:        (n_sims, T)
      infl_paths:  (n_sims, T)
    Returns two 1-D arrays of length = number of years.
    """
    # Nominal → annual
    nominal_annual = _monthly_to_annual_sum(cf_m, horizon_months)
    nominal_median_by_year = np.median(nominal_annual, axis=0)

    # Real (present-dollar) monthly → annual
    real_m = _deflate_to_present_dollars(cf_m, infl_paths)
    real_annual = _monthly_to_annual_sum(real_m, horizon_months)
    real_median_by_year = np.median(real_annual, axis=0)

    return nominal_median_by_year, real_median_by_year

