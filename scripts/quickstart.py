from portfolio_tester.config import Asset, Portfolio, SamplerConfig, SimConfig, Goal
from portfolio_tester.data.fetchers import fetch_prices_monthly, prep_returns_and_macro
from portfolio_tester.sampling.bootstrap import ReturnSampler
from portfolio_tester.engine.simulator import MonteCarloSimulator
from portfolio_tester.analytics.metrics import cagr, twrr_annualized, max_drawdown
import numpy as np

def main():
    # 1) Portfolio (MVP)
    p = Portfolio([
        Asset("VTI","Vanguard Total Stock Market ETF",0.30),
        Asset("TLT","iShares 20+ Year Treasury Bond ETF",0.40),
        Asset("IEF","iShares 7-10 Year Treasury Bond ETF",0.15),
        Asset("GSG","iShares S&P GSCI Commodity-Indexed Trust",0.075),
        Asset("GLD","SPDR Gold Shares",0.075),
    ])

    # 2) Configs
    sim_cfg = SimConfig(horizon_months=30*12, n_sims=100, starting_balance=1_000_000)  # start with 100 sims
    sam_cfg = SamplerConfig(mode="single_year", block_years=1, seed=42)

    goals = [
        # Withdraw $4,000/mo starting in 1 year, for 30 years, inflation-indexed (real)
        Goal("Retirement Withdrawals", amount=-4000, start_month=12, frequency=12, repeats=30*12, real=True),
    ]

    # 3) Data
    tickers = p.tickers()
    prices_m = fetch_prices_monthly(tickers)
    rets_m, infl_m, rf_m = prep_returns_and_macro(prices_m)

    # 4) Sample paths
    sampler = ReturnSampler(rets_m, infl_m)
    R_paths, CPI_paths = sampler.sample(sim_cfg.horizon_months, sim_cfg.n_sims, sam_cfg)

    # 5) Run simulation
    sim = MonteCarloSimulator(weights=p.weights_vector(), starting_balance=sim_cfg.starting_balance, rebalance_every_months=sim_cfg.rebalance_every_months)
    out = sim.run_with_cashflows(R_paths, CPI_paths, goals)

    # 6) Simple summary
    surv = (out["failure_month"] == -1).mean()
    cagr_vals = cagr(out["balances"], sim_cfg.horizon_months)
    twrr_vals = twrr_annualized(out["twrr_monthly"])
    mdd_vals = max_drawdown(out["balances"])

    def pct(x): return f"{100*x:.1f}%"
    print("=== Monte Carlo Summary (100 sims) ===")
    print(f"Survival rate: {pct(surv)}")
    print(f"End balance (nominal) median: ${np.median(out['balances'][:,-1]):,.0f}")
    print(f"CAGR median: {np.nanmedian(cagr_vals):.2%}")
    print(f"TWRR median: {np.nanmedian(twrr_vals):.2%}")
    print(f"Max Drawdown median: {np.median(mdd_vals):.1%}")
    print("Percentiles (10/50/90) - End Balance:",
          [f"${v:,.0f}" for v in np.percentile(out['balances'][:,-1], [10,50,90])])

    


if __name__ == "__main__":
    main()
