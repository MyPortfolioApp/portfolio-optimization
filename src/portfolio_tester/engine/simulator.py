import numpy as np
from .cashflows import build_cashflow_vector

class MonteCarloSimulator:
    def __init__(self, weights, starting_balance: float, rebalance_every_months: int = 12):
        self.w = np.array(weights, dtype=float)
        self.starting_balance = float(starting_balance)
        self.reb_m = int(rebalance_every_months)

    def run_with_cashflows(self, R_paths, infl_paths, goals):
        """
        R_paths: (n_sims, T, N) monthly simple returns per asset
        infl_paths: (n_sims, T) monthly inflation rates
        goals: list[Goal]
        Returns dict with balances, twrr_monthly, failure_month, cashflows, real_balances
        """
        n_sims, T, N = R_paths.shape
        balances = np.zeros((n_sims, T+1))
        balances[:, 0] = self.starting_balance
        failure_month = np.full(n_sims, -1, dtype=int)

        # Prebuild per-path cashflows
        cf_paths = np.zeros((n_sims, T))
        for s in range(n_sims):
            cf_paths[s] = build_cashflow_vector(goals, T, infl_path=infl_paths[s])

        alloc = np.tile(self.w, (n_sims, 1)) * self.starting_balance
        twrr_monthly = np.zeros((n_sims, T), dtype=float)

        for t in range(T):
            # 1) returns
            alloc *= (1.0 + R_paths[:, t, :])
            port = alloc.sum(axis=1)
            # time-weighted monthly pre-cashflow return
            # approximate monthly portfolio return: fixed weights dot monthly asset returns
            twrr_monthly[:, t] = (R_paths[:, t, :] * self.w).sum(axis=1)
            # 2) cashflow (end of month)
            port_after_cf = port + cf_paths[:, t]
            failed_now = (port_after_cf < 0) & (failure_month == -1)
            failure_month[failed_now] = t
            port_after_cf = np.where(port_after_cf < 0, 0.0, port_after_cf)
            balances[:, t+1] = port_after_cf

            # 3) rebalance annually
            if (t + 1) % self.reb_m == 0:
                alloc = (port_after_cf[:, None]) * self.w
            else:
                # Keep proportions from current alloc
                alloc = alloc * (port_after_cf / np.maximum(port, 1e-12))[:, None]

        # Build real (inflation-adjusted) balances per path
        real_balances = np.zeros_like(balances)
        for s in range(n_sims):
            infl_cum = np.concatenate([[1.0], np.cumprod(1.0 + infl_paths[s])])
            real_balances[s] = balances[s] / np.maximum(infl_cum, 1e-12)

        return {
            "balances": balances,
            "real_balances": real_balances,
            "twrr_monthly": twrr_monthly,
            "failure_month": failure_month,
            "cashflows": cf_paths,
            "alloc":alloc,
        }
