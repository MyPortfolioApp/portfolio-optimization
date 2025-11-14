# Portfolio Optimization

## Project Overview

With the portfolio optimization project we want to give answer to 2 main questions:
- “How robust is this specific portfolio against my plan and market uncertainty?”
- “Among all allocations that satisfy my constraints, which one has the highest chance of making it to the end of my plan without running out of money?”

On top of them, several other useful analyses are implemented to offer deeper insights and understanding to the user.

Portfolio Optimization is a small, runnable portfolio analysis and simulation framework. It uses historical data and Monte Carlo sampling to evaluate portfolio robustness, drawdowns, and survival probability under configurable goals and constraints. You can test an existing portfolio, simulate cashflows (contributions/withdrawals), and optionally search a constrained frontier for the allocation that maximizes survival rate for your goal.

The project includes a minimal quickstart script and a survival‑optimization script that demonstrate end‑to‑end runs, metrics, and plots.

## Key Features

- Historical-bootstrap Monte Carlo of monthly returns (single-month, single-year, and k‑year block sampling modes).
- Configurable contributions/withdrawals (“goals”) with timing, frequency (monthly/quarterly/annual), repeats, and inflation indexing (real vs nominal).
- Survival calculation (share of simulated paths with positive end-of-horizon wealth); simulator also reports the first failure month per path.
- Risk/return metrics: CAGR, time‑weighted return (TWRR), max drawdown; Sharpe/Sortino helpers on historical series.
- Drawdown analysis, including “including cashflows” vs “excluding cashflows (TWRR index)” variants.
- Efficient frontier and max‑Sharpe portfolio (long‑only), with transition map of weights across the frontier.
- Visualization: allocation donut, percentile bands, end balance histogram, survival curve, drawdown histograms, correlation matrix, and returns/risk table.
- Constrained frontier builder + survival optimizer: evaluate survival along a frontier, smooth, and pick the best‑survival allocation; refine the winner with more simulations.
- Data utilities: Yahoo Finance prices (cached), CPI and 3‑month T‑bill (FRED) aligned to returns; local caching in `data_cache/`.

## Installation

Requirements
- Python 3.10+ (PEP 604 union types and annotations are used).
- Dependencies: `pip install -r requirements.txt`.

Create a virtual environment
- macOS / Linux (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- Windows (PowerShell):
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies
```bash
pip install -r requirements.txt
```

Notes and troubleshooting
- Network access: price data is fetched from Yahoo Finance (via `yfinance`) and CPI/T‑bill from FRED. Data is cached under `data_cache/` to avoid re‑downloads.
- Proxies: FRED fetching uses `requests` with `trust_env=False` and `proxies={"http": None, "https": None}` to bypass system proxies. If you rely on proxies, you may need to adapt `fetch_fred_series` in `src/portfolio_tester/data/fetchers.py`.
- API keys: none required for Yahoo or FRED in this setup.

## Configuration and Inputs

Configurations use simple Python dataclasses and function arguments. The typical flow mirrors `scripts/quickstart.py`:

```python
from portfolio_tester.config import Asset, Portfolio, SimConfig, SamplerConfig, Goal
from portfolio_tester.data.fetchers import fetch_prices_monthly, prep_returns_and_macro
from portfolio_tester.sampling.bootstrap import ReturnSampler
from portfolio_tester.engine.simulator import MonteCarloSimulator

# 1) Portfolio (weights sum to 1)
p = Portfolio([
    Asset("VTI", "Vanguard Total Stock Market ETF", 0.30),
    Asset("TLT", "iShares 20+ Year Treasury Bond ETF", 0.40),
    Asset("IEF", "iShares 7-10 Year Treasury Bond ETF", 0.15),
    Asset("GSG", "iShares S&P GSCI Commodity-Indexed Trust", 0.075),
    Asset("GLD", "SPDR Gold Shares", 0.075),
])

# 2) Simulation + sampling
sim_cfg = SimConfig(horizon_months=30*12, n_sims=1000, rebalance_every_months=12, starting_balance=1_000_000.0)
sam_cfg = SamplerConfig(mode="single_year", block_years=1, seed=42)

# 3) Cashflow goals (negative = withdrawal; positive = contribution)
goals = [
    Goal("Retirement Withdrawals", amount=-4000, start_month=12, frequency=12, repeats=30*12, real=True),
]

# 4) Data and macro series
prices_m = fetch_prices_monthly(p.tickers())  # optional start/end supported
rets_m, infl_m, rf_m = prep_returns_and_macro(prices_m)

# 5) Sample paths and run Monte Carlo
sampler = ReturnSampler(rets_m, infl_m)
R_paths, CPI_paths = sampler.sample(sim_cfg.horizon_months, sim_cfg.n_sims, sam_cfg)
sim = MonteCarloSimulator(weights=p.weights_vector(), starting_balance=sim_cfg.starting_balance, rebalance_every_months=sim_cfg.rebalance_every_months)
out = sim.run_with_cashflows(R_paths, CPI_paths, goals)
```

Inputs and parameters
- Assets/tickers and initial weights: `Portfolio` of `Asset` entries.
- Historical data source: `fetch_prices_monthly(tickers, start=None, end=None)` from Yahoo Finance; cached locally.
- Macro series: `prep_returns_and_macro(prices)` produces monthly returns, CPI inflation (as monthly rate), and monthly risk‑free rate (TB3MS converted to a monthly compound rate).
- Simulation: `SimConfig(horizon_months, n_sims, rebalance_every_months, starting_balance)`.
- Sampling: `SamplerConfig(mode, block_years, seed)` with modes:
  - `"single_month"`: bootstrap individual months (iid sampling across months).
  - `"single_year"`: resample entire 12‑month years.
  - `"block_years"`: resample contiguous k‑year blocks (rolling around the available years).
- Inflation: CPI is used for “real” cashflows (indexing amounts to the payment date) and to produce “real” balance paths.

## Goals and Cashflow Settings

Goals model cashflows applied at end‑of‑month.

`Goal` fields
- `name`: label for the cashflow stream.
- `amount`: positive = contribution (deposit), negative = withdrawal.
- `start_month`: first payment offset from t=0 (e.g., 12 means 1 year from start).
- `frequency`: payments per year: `1` (annual), `4` (quarterly), `12` (monthly).
- `repeats`: number of payments.
- `real`: if `True`, the nominal payment amount is indexed by sampled CPI to the payment date (present‑dollar amount kept constant).

Examples
1) Retirement‑style withdrawals (monthly, inflation‑indexed):
```python
goals = [Goal("Retirement Withdrawals", amount=-4000, start_month=12, frequency=12, repeats=30*12, real=True)]
```
2) Accumulation with contributions (monthly, nominal):
```python
goals = [Goal("Contributions", amount=+1000, start_month=0, frequency=12, repeats=10*12, real=False)]
```
3) Lump‑sum at a specific time (e.g., one‑time withdrawal in 5 years):
```python
goals = [Goal("College", amount=-50000, start_month=5*12, frequency=1, repeats=1, real=True)]
```

How goals affect survival
- Simulator tracks the first month a path becomes negative after a cashflow (`failure_month`), then clamps balance to 0 for subsequent months.
- Survival rate used in optimization is “% of paths with positive ending wealth.” The simulator also exposes `failure_month` so you can analyze the timing of failures and draw survival curves.

## Core Concepts and Outputs

Outputs from `MonteCarloSimulator.run_with_cashflows(...)`:
- `balances`: `(n_sims, T+1)` nominal balances including cashflows.
- `real_balances`: `(n_sims, T+1)` CPI‑deflated balances (present dollars).
- `twrr_monthly`: `(n_sims, T)` monthly time‑weighted returns of the portfolio (pre‑cashflows); useful for drawdowns excluding cashflows.
- `failure_month`: `(n_sims,)` first failure month index or `-1` if never failed.
- `cashflows`: `(n_sims, T)` cashflow series applied during the simulation.

Key metrics and plots
- Survival rate: fraction of paths with positive ending balance; survival curve from `failure_month` shows probability of no failure through time.
- End wealth distribution: median and percentiles of `balances[:, -1]`.
- Drawdowns: `max_drawdown` on `balances` (includes cashflows) and on a TWRR index (excludes cashflows).
- Risk/return: CAGR (`cagr`), annualized TWRR (`twrr_annualized`); Sharpe/Sortino helpers on historical monthly returns.
- Plots (saved by scripts): allocation donut, efficient frontier and transition map, percentile bands (nominal/real), end balance histogram, survival curve, simulated cashflows, drawdown histograms, correlation matrix, and returns/risk table.

Standard output locations
- Figures: `figures/` for quickstart; `figures/survival_optimizer/` for the optimization script.
- Data cache: `data_cache/` (CSV files keyed by request).

## Codebase Structure

```
src/portfolio_tester/
  config.py                # Asset, Portfolio, Goal, DataConfig, SamplerConfig, SimConfig
  sampling/bootstrap.py    # ReturnSampler with single_month / single_year / block_years
  engine/
    simulator.py           # Monte Carlo engine, rebalancing, cashflow application
    cashflows.py           # Build monthly cashflow vectors; annual cashflow medians
    optimizer.py           # Constrained frontier + survival evaluation/selection
  analytics/
    metrics.py             # CAGR, TWRR, IRR, drawdown helpers
    risk.py                # Mean/cov, efficient frontier, max Sharpe, asset stats
  data/
    fetchers.py            # Yahoo prices; FRED CPI/TB3MS; return prep; caching hooks
    cache.py               # CSV cache in data_cache/
  viz/charts.py            # Plot helpers: frontier, bands, histograms, tables, etc.

scripts/
  quickstart.py            # Minimal end‑to‑end simulation + plots for one portfolio
  optimize_survival.py     # Build constrained frontier, compute survival, pick best

notebooks/ (various)       # Exploratory notebooks (if present)
requirements.txt           # Python dependencies
pyproject.toml             # Package metadata (source in src/)
```

## Usage Examples

Quickstart (single portfolio)
```bash
python scripts/quickstart.py
```
What it does
- Downloads monthly prices and macro series (cached), runs a Monte Carlo over the configured horizon with cashflows, prints summary metrics, and saves figures to `figures/`.

Customize basics (edit `scripts/quickstart.py`)
- Change tickers and weights in `Portfolio([...])`.
- Adjust `SimConfig(horizon_months, n_sims, rebalance_every_months, starting_balance)`.
- Pick sampling mode in `SamplerConfig(mode=...)` (`single_month`, `single_year`, `block_years`).
- Modify `goals` to set contributions/withdrawals, frequency, repeats, and real vs nominal.

Survival‑optimized portfolio
```bash
python scripts/optimize_survival.py
```
What it does
- Builds a constrained mean‑variance frontier, evaluates survival for each portfolio via Monte Carlo, smooths survival, selects the best, re‑simulates that portfolio with more paths, and saves summary plots under `figures/survival_optimizer/`.

Where to tweak
- Constraints: see `OptimizationConstraints` in `src/portfolio_tester/engine/optimizer.py` and the example in `scripts/optimize_survival.py` (per‑asset min/max weights, allow_short, optional volatility cap).
- Horizon/sims/sampling/goals: same as quickstart (`SimConfig`, `SamplerConfig`, `Goal`).

## Extending the Project

- New data source or transformations
  - Add fetchers under `src/portfolio_tester/data/` and wire into `prep_returns_and_macro`. Keep outputs aligned monthly with returns.
  - Respect `data_cache/` for caching large downloads.

- New goal/cashflow patterns
  - Extend `build_cashflow_vector` in `engine/cashflows.py` or introduce a richer goal model and adapt the builder. Keep sign conventions (+contribution, −withdrawal) and the end‑of‑month application consistent.

- New analysis/metrics/plots
  - Add to `analytics/` or `viz/` and call them from scripts. Follow the existing function signatures and plotting style for consistency.

- New sampling methods
  - Extend `ReturnSampler` in `sampling/bootstrap.py` and add a new `SamplerConfig.mode` option if needed.

Key abstractions
- Configuration: `Asset`, `Portfolio`, `Goal`, `SimConfig`, `SamplerConfig`.
- Engines: `ReturnSampler`, `MonteCarloSimulator`.
- Optimization: frontier building + survival evaluation (`engine/optimizer.py`).
- Results: simulator returns rich arrays in `out` for downstream analysis/plotting.

## License and Credits

- License: No explicit license file is included. If you plan to use this beyond personal/educational purposes, please contact the author or add a LICENSE file.
- Data acknowledgements: Yahoo Finance (prices via `yfinance`), Federal Reserve Bank of St. Louis FRED (CPI `CPIAUCSL`, T‑bill `TB3MS`).
- Libraries: NumPy, pandas, SciPy, Matplotlib, yfinance, requests, numpy‑financial.
