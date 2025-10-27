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
