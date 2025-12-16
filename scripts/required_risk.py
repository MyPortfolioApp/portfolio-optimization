"""
Calculate the required annual rate of return for a portfolio to survive,
meaning it has enough funds to cover all withdrawals (goals with negative amounts).
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import brentq

from portfolio_tester.config import Goal
from portfolio_tester.engine.cashflows import build_cashflow_vector


def find_required_annual_return(
    goals: List[Goal],
    starting_balance: float,
    horizon_months: int | None = None,
    annual_inflation: float = 0.015,
    tolerance: float = 1e-8,
) -> float | None:
    """
    Find the minimum annual rate of return required for a portfolio to survive.

    A portfolio "survives" if it never runs out of money to cover all withdrawals
    (goals with negative amounts) over the investment horizon.

    The function uses numerical root-finding (Brent's method) to solve for the
    annual return rate `r` such that the final portfolio balance is exactly zero.

    Parameters
    ----------
    goals : List[Goal]
        A list of Goal objects representing contributions (positive amounts)
        and withdrawals (negative amounts).
    starting_balance : float
        The initial portfolio balance.
    horizon_months : int, optional
        The investment horizon in months. If not provided, it is inferred from
        the goals as the last month with a scheduled cash flow.
    annual_inflation : float, optional
        Annual inflation rate used to adjust cashflows for goals with real=True.
        Default is 0.015 (1.5% per year).
    tolerance : float, optional
        Numerical tolerance for the root-finding algorithm. Default is 1e-8.

    Returns
    -------
    float or None
        The required annual rate of return as a decimal (e.g., 0.05 for 5%,
        or -0.02 for -2%). Can be negative if the portfolio has net positive
        cashflows and would end with excess funds at 0% return.
        Returns None if no valid rate can be found within reasonable bounds.

    Notes
    -----
    - The function assumes monthly compounding.
    - For goals with real=True, cashflows are inflation-adjusted using the
      specified annual_inflation rate (default 1.5%).


    Examples
    --------
    >>> from portfolio_tester.config import Goal
    >>> goals = [
    ...     Goal("Withdrawal", amount=-5000, start_month=12, frequency=12, repeats=20, real=False),
    ... ]
    >>> rate = find_required_annual_return(goals, starting_balance=100_000)
    >>> print(f"Required annual return: {rate:.2%}")
    """
    # Infer horizon from goals if not provided
    if horizon_months is None:
        horizon_months = _infer_horizon(goals)

    if horizon_months <= 0:
        return None

    # Build inflation path for real cashflows (fixed annual rate -> monthly)
    monthly_inflation = (1.0 + annual_inflation) ** (1 / 12) - 1
    infl_path = np.full(horizon_months, monthly_inflation)

    # Build the cashflow vector with inflation adjustment for real goals
    cashflows = build_cashflow_vector(goals, horizon_months, infl_path=infl_path)

    def final_balance(annual_rate: float) -> float:
        """Compute final balance given an annual return rate."""
        monthly_rate = (1.0 + annual_rate) ** (1 / 12) - 1
        balance = starting_balance
        for month in range(horizon_months):
            # Apply monthly growth
            balance *= 1.0 + monthly_rate
            # Apply end-of-month cashflow
            balance += cashflows[month]
        return balance

    # Check boundary conditions
    balance_at_zero = final_balance(0.0)
    
    # Determine search bounds based on whether we need positive or negative returns
    if balance_at_zero > 0:
        # Portfolio has excess funds at 0% return, need negative return to reach zero
        lower = -0.99  # -99% annual return (portfolio loses almost everything)
        upper = 0.0
        
        # Find lower bound where final balance becomes negative
        while final_balance(lower) > 0:
            lower = lower * 2 if lower < -0.5 else lower - 0.1
            if lower < -0.999:  # -99.9% annual return - unrealistic
                return None
    elif balance_at_zero < 0:
        # Portfolio runs out of money at 0% return, need positive return
        lower = 0.0
        upper = 0.5  # 50% annual return as initial upper bound
        
        # Find upper bound where portfolio survives
        while final_balance(upper) < 0:
            upper *= 2
            if upper > 100:  # 10,000% annual return - unrealistic
                return None
    else:
        # Exactly zero at 0% return
        return 0.0

    # Use Brent's method to find the root
    try:
        required_rate = brentq(final_balance, lower, upper, xtol=tolerance)
        return required_rate
    except ValueError:
        return None


def _infer_horizon(goals: List[Goal]) -> int:
    """Infer the investment horizon from the last scheduled cash flow."""
    if not goals:
        return 0

    max_month = 0
    for g in goals:
        step = 12 // g.frequency  # months between payments
        last_month = g.start_month + step * (g.repeats - 1)
        max_month = max(max_month, last_month)

    # Add 1 because months are 0-indexed and we need to include the last month
    return max_month + 1


# --- Example usage ---
if __name__ == "__main__":
    # Example: Retirement scenario with inflation-adjusted withdrawals
    # Starting with $1,000,000, withdrawing $50,000/year (in today's dollars) for 30 years
    starting_balance = 1_000_000

    goals = [
        Goal(
            "Retirement Withdrawals",
            amount=-50_000,
            start_month=12,
            frequency=1,  # annual withdrawals
            repeats=30,
            real=True,  # Inflation-adjusted at 1.5%/year
        ),
    ]

    required_rate = find_required_annual_return(goals, starting_balance)

    if required_rate is not None:
        print(f"Starting balance: ${starting_balance:,.0f}")
        print(f"Withdrawals: $50,000/year (real, inflation-adjusted at 1.5%) for 30 years")
        print(f"Required annual return: {required_rate:.2%}")
    else:
        print("Could not determine required return rate.")

    # Compare with nominal (non-inflation-adjusted) withdrawals
    print("\n--- Comparison: Nominal vs Real withdrawals ---")
    goals_nominal = [
        Goal("Retirement", amount=-50_000, start_month=12, frequency=1, repeats=30, real=False),
    ]
    rate_nominal = find_required_annual_return(goals_nominal, starting_balance)
    print(f"Nominal withdrawals ($50k fixed): {rate_nominal:.2%} required")
    print(f"Real withdrawals ($50k + 1.5% inflation): {required_rate:.2%} required")

    # Another example with contributions
    print("\n--- Example with mixed cashflows ---")
    goals_mixed = [
        Goal("Salary", amount=+2_000, start_month=1, frequency=12, repeats=120, real=False),  # 10 years of income
        Goal("Retirement", amount=-4_000, start_month=121, frequency=12, repeats=240, real=False),  # 20 years retirement
    ]

    required_rate_mixed = find_required_annual_return(goals_mixed, starting_balance=50_000)
    if required_rate_mixed is not None:
        print(f"Required annual return: {required_rate_mixed:.2%}")
    else:
        print("Could not determine required return rate.")

    # Example with net positive cashflows (contributions > withdrawals)
    print("\n--- Example with net positive cashflows (negative required return) ---")
    goals_positive = [
        Goal("Large Inheritance", amount=+500_000, start_month=60, frequency=1, repeats=1, real=False),  # One-time
        Goal("Small Withdrawals", amount=-10_000, start_month=12, frequency=1, repeats=20, real=False),  # $10k/year for 20 years
    ]

    required_rate_positive = find_required_annual_return(goals_positive, starting_balance=100_000)
    if required_rate_positive is not None:
        print(f"Starting balance: $100,000")
        print(f"Inheritance in 5 years: $500,000")
        print(f"Withdrawals: $10,000/year for 20 years (total: $200,000)")
        print(f"Required annual return: {required_rate_positive:.2%}")
        if required_rate_positive < 0:
            print(f"  → Portfolio can afford to LOSE {abs(required_rate_positive):.2%} per year and still break even!")
    else:
        print("Could not determine required return rate.")
