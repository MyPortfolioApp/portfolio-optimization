from portfolio_tester.config import Asset, Portfolio, SamplerConfig, SimConfig, Goal
from portfolio_tester.data.fetchers import fetch_prices_monthly, prep_returns_and_macro, fetch_fred_series
from portfolio_tester.sampling.bootstrap import ReturnSampler
from portfolio_tester.engine.simulator import MonteCarloSimulator
from portfolio_tester.analytics.metrics import cagr, twrr_annualized, max_drawdown, sharpe_sortino
from portfolio_tester.analytics.risk import efficient_frontier, portfolio_annual_stats, single_asset_stats, max_sharpe_portfolio, risk_free_annual
from portfolio_tester.viz.charts import plot_allocation_donut, plot_efficient_frontier, plot_end_balance_hist, plot_percentile_bands, plot_survival_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    figs_dir = Path("figures"); figs_dir.mkdir(exist_ok=True, parents=True)

    # 1) Portfolio (MVP)
    p = Portfolio([
        Asset("VTI","Vanguard Total Stock Market ETF",0.30),
        Asset("TLT","iShares 20+ Year Treasury Bond ETF",0.40),
        Asset("IEF","iShares 7-10 Year Treasury Bond ETF",0.15),
        Asset("GSG","iShares S&P GSCI Commodity-Indexed Trust",0.075),
        Asset("GLD","SPDR Gold Shares",0.075),
    ])

    # 2) Configs
    sim_cfg = SimConfig(horizon_months=30*12, n_sims=100, starting_balance=1_000_000.0)
    sam_cfg = SamplerConfig(mode="single_year", block_years=1, seed=42)

    # Withdraw $4,000/mo starting in 1 year, for 30 years, inflation-indexed (real)
    goals = [
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

    # 6) Metrics
    surv_rate = (out["failure_month"] == -1).mean()
    cagr_vals = cagr(out["balances"], sim_cfg.horizon_months)
    twrr_vals = twrr_annualized(out["twrr_monthly"])
    mdd_vals = max_drawdown(out["balances"])
    #sharpe_vals, sortino_vals = sharpe_sortino(out["twrr_monthly"], rf_m) This should be calculated on past metrics not simulated paths

    def pct(x): return f"{100*x:.1f}%"
    print("=== Monte Carlo Summary ===")
    print(f"Survival rate: {pct(surv_rate)}")
    print(f"End balance (nominal) median: ${np.median(out['balances'][:,-1]):,.0f}")
    print(f"CAGR median: {np.nanmedian(cagr_vals):.2%}")
    print(f"TWRR median: {np.nanmedian(twrr_vals):.2%}")
    #print(f"Sharpe (median): {np.nanmedian(sharpe_vals):.2f} | Sortino (median): {np.nanmedian(sortino_vals):.2f}")
    print(f"Max Drawdown median: {np.median(mdd_vals):.1%}")
    print("Percentiles (10/50/90) - End Balance:",
          [f"${v:,.0f}" for v in np.percentile(out['balances'][:,-1], [10,50,90])])

    # 7) Figures
    labels = [a.name for a in p.assets]
    weights = p.weights_vector()
    fig, ax = plot_allocation_donut(labels, weights, title="MVP Portfolio Allocation !")
    fig.savefig(figs_dir / "allocation_donut.png", bbox_inches="tight"); plt.close(fig)

    # Efficient frontier (upper branch only)
    W, rets, risks, names = efficient_frontier(rets_m)
    # Current portfolio stats
    pr, pv = portfolio_annual_stats(weights, rets_m)
    # Single-asset points
    asset_sigma, asset_mu, asset_labels = single_asset_stats(rets_m)
    # Max Sharpe portfolio (long-only) + risk-free (annualized)
    w_msr, r_msr, v_msr, sh_msr, rf_a, _ = max_sharpe_portfolio(rets_m, rf_m)
    fig, ax = plot_efficient_frontier(
        risks, rets,
        port_pt=(pv, pr),
        asset_points=(asset_sigma, asset_mu, asset_labels),
        msr_pt=(v_msr, r_msr),
        rf=rf_a,
        port_label="Provided Portfolio",
        title="Efficient Frontier (upper branch) with Max Sharpe"
    )
    fig.savefig(figs_dir / "efficient_frontier.png", bbox_inches="tight"); plt.close(fig)
    print(f"Max Sharpe (annual): return={r_msr:.2%}, vol={v_msr:.2%}, Sharpe={(r_msr - rf_a)/max(v_msr,1e-16):.2f}")


    fig, ax = plot_percentile_bands(out["balances"], title="Nominal Balance Percentile Bands")
    fig.savefig(figs_dir / "bands_nominal.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_percentile_bands(out["real_balances"], title="Real (CPI-adjusted) Balance Percentile Bands")
    fig.savefig(figs_dir / "bands_real.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_end_balance_hist(out["balances"][:,-1], title="Ending Balance (Nominal)")
    fig.savefig(figs_dir / "end_balance_hist.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_survival_curve(out["failure_month"], sim_cfg.horizon_months, title="Survival Curve (No Failure by Month)")
    fig.savefig(figs_dir / "survival_curve.png", bbox_inches="tight"); plt.close(fig)

    print(f"Saved figures to: {figs_dir.resolve()}")

if __name__ == "__main__":
    main()
