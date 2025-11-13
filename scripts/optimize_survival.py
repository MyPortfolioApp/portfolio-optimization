from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from portfolio_tester.analytics.metrics import cagr, twrr_annualized, max_drawdown
from portfolio_tester.analytics.risk import (
    efficient_frontier,
    portfolio_annual_stats,
    single_asset_stats,
    max_sharpe_portfolio,
)
from portfolio_tester.config import Asset, Goal, Portfolio, SamplerConfig, SimConfig
from portfolio_tester.data.fetchers import fetch_prices_monthly, prep_returns_and_macro
from portfolio_tester.engine.cashflows import annual_cashflow_medians
from portfolio_tester.engine.optimizer import (
    OptimizationConstraints,
    build_constrained_frontier,
    compute_survival_on_frontier,
    find_best_survival_portfolio,
    refine_best_portfolio,
)
from portfolio_tester.sampling.bootstrap import ReturnSampler
from portfolio_tester.viz.charts import (
    plot_allocation_donut,
    plot_efficient_frontier,
    plot_percentile_bands,
    plot_end_balance_hist,
    plot_survival_curve,
    plot_simulated_annual_cashflows,
    plot_max_drawdown_histograms,
    plot_frontier_transition_map,
    plot_correlation_matrix,
    plot_returns_risk_table,
    plot_survival_vs_vol
)





def main():
    figs_dir = Path("figures") / "survival_optimizer"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Portfolio universe and goals (mirroring quickstart defaults)
    portfolio = Portfolio(
        [
            Asset("VTI", "Vanguard Total Stock Market ETF", 0.30),
            Asset("TLT", "iShares 20+ Year Treasury Bond ETF", 0.40),
            Asset("IEF", "iShares 7-10 Year Treasury Bond ETF", 0.15),
            Asset("GSG", "iShares S&P GSCI Commodity-Indexed Trust", 0.075),
            Asset("GLD", "SPDR Gold Shares", 0.075),
        ]
    )
    tickers = portfolio.tickers()
    asset_labels = [a.name for a in portfolio.assets]

    sim_cfg = SimConfig(horizon_months=30 * 12, n_sims=1000, rebalance_every_months=12, starting_balance=1_000_000.0)
    sampler_cfg = SamplerConfig(mode="single_year", block_years=1, seed=42)
    
    
    goals = [
        Goal("Retirement Withdrawals", amount=-4000, start_month=12, frequency=12, repeats=30 * 12, real=True),
    ]

    # Data + sampler
    prices_m = fetch_prices_monthly(tickers)
    rets_m, infl_m, rf_m = prep_returns_and_macro(prices_m)
    sampler = ReturnSampler(rets_m, infl_m)

    # Optimization settings
    constraints = OptimizationConstraints(
        min_weights=[0.05, 0.05, 0.05, 0.0, 0.0],
        max_weights=[0.55, 0.55, 0.40, 0.25, 0.25],
        allow_short=False,
    )
    frontier = build_constrained_frontier(rets_m, constraints, n_portfolios=10)
    compute_survival_on_frontier(
        frontier,
        sim_config=sim_cfg,
        sampler=sampler,
        goals=goals,
        sampler_config=sampler_cfg,
        n_sims_per_portfolio=100,
    )
    result = find_best_survival_portfolio(frontier, smoothing_window=5)
    refine_best_portfolio(
        result,
        sim_config=sim_cfg,
        sampler=sampler,
        goals=goals,
        sampler_config=sampler_cfg,
        n_sims_best=100,
    )
    if result.best_simulation_output is None:
        raise RuntimeError("Refinement step did not return simulation output.")

    best_fp = result.best_portfolio
    best_weights = best_fp.weights
    best_out = result.best_simulation_output

    # Monte Carlo stats for the best-survival portfolio
    nominal_median_by_year, real_median_by_year = annual_cashflow_medians(
        best_out["cashflows"],
        best_out["infl_paths"],
        sim_cfg.horizon_months,
    )
    cagr_vals = cagr(best_out["balances"], sim_cfg.horizon_months)
    twrr_vals = twrr_annualized(best_out["twrr_monthly"])
    mdd_including = max_drawdown(best_out["balances"])
    twrr = best_out["twrr_monthly"]
    idx_ex_cash = np.concatenate(
        [np.ones((twrr.shape[0], 1)), np.cumprod(1.0 + twrr, axis=1)],
        axis=1,
    )
    mdd_excluding = max_drawdown(idx_ex_cash)

    best_ret, best_vol = portfolio_annual_stats(best_weights, rets_m)

    def pct(x: float) -> str:
        return f"{100 * x:.2f}%"

    print("=== Survival-Optimized Portfolio ===")
    for ticker, label, weight in zip(tickers, asset_labels, best_weights):
        print(f"{ticker:<5} | {label:<40} | weight={pct(weight)}")
    print(f"\nExpected return: {best_ret:.2%}")
    print(f"Volatility: {best_vol:.2%}")
    print(f"Raw survival rate: {pct(best_fp.survival_rate or 0.0)}")
    if best_fp.smoothed_survival is not None:
        print(f"Smoothed survival at optimum: {pct(best_fp.smoothed_survival)}")
    print(f"Median end balance: ${np.median(best_out['balances'][:, -1]):,.0f}")
    print(f"CAGR median: {np.nanmedian(cagr_vals):.2%}")
    print(f"TWRR median: {np.nanmedian(twrr_vals):.2%}")
    print(f"Max Drawdown (incl cashflows) median: {np.median(mdd_including):.1%}")

    # --- Visualization outputs ---
    labels = asset_labels
    fig, ax = plot_allocation_donut(labels, best_weights, title="Best Survival Portfolio Allocation")
    fig.savefig(figs_dir / "allocation_donut.png", bbox_inches="tight"); plt.close(fig)

    _, rets, risks, _ = efficient_frontier(rets_m)
    asset_sigma, asset_mu, single_labels = single_asset_stats(rets_m)
    _, r_msr, v_msr, _, rf_a, _ = max_sharpe_portfolio(rets_m, rf_m)
    fig, ax = plot_efficient_frontier(
        risks,
        rets,
        port_pt=(best_vol, best_ret),
        asset_points=(asset_sigma, asset_mu, single_labels),
        msr_pt=(v_msr, r_msr),
        rf=rf_a,
        port_label="Best Survival Portfolio",
        title="Efficient Frontier with Survival-Optimized Portfolio",
    )
    fig.savefig(figs_dir / "efficient_frontier.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_frontier_transition_map(
        weights=np.array([fp.weights for fp in result.frontier]),
        risks=np.array([fp.volatility for fp in result.frontier]),
        asset_labels=asset_labels,
        port_risk=best_vol,
        msr_risk=v_msr,
        title=f"Constrained Frontier Transition Map ({rets_m.index[0]:%b %Y} - {rets_m.index[-1]:%b %Y})",
    )
    fig.savefig(figs_dir / "constrained_transition.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_survival_vs_vol(
        frontier=result.frontier,
        smoothed_rates=result.smoothed_survival_rates,
        best_idx=result.best_index,
    )
    fig.savefig(figs_dir / "survival_vs_vol.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_percentile_bands(best_out["balances"], title="Nominal Balance Percentile Bands")
    fig.savefig(figs_dir / "bands_nominal.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_percentile_bands(best_out["real_balances"], title="Real Balance Percentile Bands")
    fig.savefig(figs_dir / "bands_real.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_end_balance_hist(best_out["balances"][:, -1], title="Ending Balance Distribution")
    fig.savefig(figs_dir / "ending_balance_hist.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plot_survival_curve(best_out["failure_month"], sim_cfg.horizon_months, title="Survival Curve (Best Portfolio)")
    fig.savefig(figs_dir / "survival_curve.png", bbox_inches="tight"); plt.close(fig)

    fig, _ = plot_simulated_annual_cashflows(
        nominal_median_by_year,
        real_median_by_year,
        title_top="Simulated Annual Cashflows (Nominal)",
        title_bottom="Simulated Annual Cashflows (Real)",
    )
    fig.savefig(figs_dir / "simulated_cashflows.png", bbox_inches="tight"); plt.close(fig)

    fig, _ = plot_max_drawdown_histograms(
        mdd_including=mdd_including,
        mdd_excluding=mdd_excluding,
        bins=20,
        clip_to_pct=95,
    )
    fig.savefig(figs_dir / "max_drawdown_histograms.png", bbox_inches="tight"); plt.close(fig)

    rets_for_tables = rets_m.copy()
    name_map = {a.ticker: a.name for a in portfolio.assets}
    table_labels = [name_map.get(col, col) for col in rets_for_tables.columns]
    period = f"{rets_for_tables.index[0]:%b %Y} to {rets_for_tables.index[-1]:%b %Y}"
    subtitle = f"Statistics based on monthly returns from {period}."

    fig, _ = plot_correlation_matrix(
        returns_m=rets_for_tables,
        asset_labels=table_labels,
        title="Asset Correlation Matrix",
        subtitle=subtitle,
    )
    fig.savefig(figs_dir / "asset_correlation_matrix.png", bbox_inches="tight"); plt.close(fig)

    fig, _ = plot_returns_risk_table(
        returns_m=rets_for_tables,
        asset_labels=table_labels,
        title="Asset Returns & Risk (Annualized)",
        subtitle=subtitle,
    )
    fig.savefig(figs_dir / "asset_returns_risk_table.png", bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved figures to: {figs_dir.resolve()}")


if __name__ == "__main__":
    main()
