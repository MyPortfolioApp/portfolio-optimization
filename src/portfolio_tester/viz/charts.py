import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
from matplotlib import gridspec
import matplotlib as mpl
import pandas as pd


def _ensure_ax(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    else:
        fig = ax.figure
    return fig, ax

def plot_allocation_donut(labels: Sequence[str], weights: Sequence[float], ax=None, title="Portfolio Allocation"):
    """Donut chart showing allocation percentages on wedges and in the legend."""

    fig, ax = _ensure_ax(ax)
    weights = np.asarray(list(weights), dtype=float)
    wnorm = weights / weights.sum() if weights.sum() != 0 else weights

    # Show % labels on wedges; hide very small ones to avoid clutter
    def _autopct(pct, threshold=2.0):
        return f"{pct:.1f}%" if pct >= threshold else ""

    wedges, _, _  = ax.pie(
        weights,
        startangle=90,
        wedgeprops=dict(width=0.4),
        autopct=lambda pct: _autopct(pct),
        pctdistance=0.8,  # place the % text inside the ring
    )

    # Legend also shows percentages, normalized to sum=100%
    legend_labels = [f"{lab} — {p*100:.1f}%" for lab, p in zip(labels, wnorm)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    return fig, ax

def plot_efficient_frontier(
    risks: np.ndarray,
    rets: np.ndarray,
    port_pt: Tuple[float, float] = None,
    asset_points: Tuple[np.ndarray, np.ndarray, Sequence[str]] = None,
    msr_pt: Tuple[float, float] = None,
    rf: float = None,
    port_label: str = "Provided Portfolio",
    ax=None,
    title: str = "Efficient Frontier"
):
    fig, ax = _ensure_ax(ax)

    # Efficient (upper) frontier
    ax.plot(risks, rets, lw=2, label="Efficient frontier")

    # Single-asset markers + labels
    if asset_points is not None:
        a_risk, a_ret, a_labels = asset_points
        ax.scatter(a_risk, a_ret, s=60)
        for x, y, lab in zip(a_risk, a_ret, a_labels):
            ax.annotate(lab, xy=(x, y), xytext=(6, 6), textcoords="offset points")

    # Provided portfolio
    if port_pt is not None:
        ax.scatter([port_pt[0]], [port_pt[1]], s=60, marker="D", label=port_label)
        ax.annotate(port_label, xy=port_pt, xytext=(8, 8), textcoords="offset points")

    # Max Sharpe portfolio (star) + Capital Market Line
    if msr_pt is not None:
        ax.scatter([msr_pt[0]], [msr_pt[1]], s=160, marker="*", label="Max Sharpe")
        ax.annotate("Max Sharpe", xy=msr_pt, xytext=(8, 8), textcoords="offset points")
        if rf is not None:
            # CML from (0, rf) to the MSR point
            ax.plot([0.0, msr_pt[0]], [rf, msr_pt[1]], linestyle="--", linewidth=1.5, label="Capital Market Line")

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig, ax


def plot_frontier_transition_map(
    weights: np.ndarray,          # (n_pts, n_assets) from efficient_frontier(...)
    risks: np.ndarray,            # (n_pts,)
    asset_labels: Sequence[str],
    port_risk: float | None = None,   # <-- NEW
    msr_risk: float | None = None,    # <-- NEW
    ax=None,
    title: str = "Efficient Frontier Transition Map"
):
    """
    Stacked area (stackplot) of long-only frontier weights vs. portfolio volatility.
    Shows how the optimal allocation transitions across the efficient frontier.
    Optionally draws vertical lines at the Provided Portfolio risk and Max Sharpe risk.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = _ensure_ax(ax)

    # 1) sort by risk (x-axis)
    order = np.argsort(risks)
    x = risks[order].astype(float)

    # 2) clip tiny negatives from numerical noise and re-normalize to 1
    W = np.clip(weights[order, :].astype(float), 0.0, None)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    # 3) stackplot expects one series per asset; transpose and convert to %
    Y = (W.T) * 100.0

    # 4) draw the stacked area
    ax.stackplot(x, Y, labels=asset_labels, alpha=0.5)

    # 5) top border (cosmetic)
    ax.plot([x.min(), x.max()], [100.0, 100.0], linewidth=1, alpha=0.4)

    # 6) optional vertical lines + tiny labels above the stack
    if port_risk is not None:
        ax.axvline(port_risk, color="k", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.annotate("Provided Portfolio", xy=(port_risk, 75),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom")

    if msr_risk is not None:
        ax.axvline(msr_risk, color="k", linestyle=":", linewidth=1.8, alpha=0.9)
        ax.annotate("Max Sharpe", xy=(msr_risk, 50),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom")

    # 7) labels, scales, legend
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Allocation")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}%"))
    ax.grid(True, alpha=0.2)
    ax.set_title(title)

    # legend below chart
    ncol = min(len(asset_labels), 4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False)

    return fig, ax





def plot_end_balance_hist(end_balances, ax=None, title="End Balance Distribution"):
    fig, ax = _ensure_ax(ax)
    ax.hist(end_balances, bins=40)
    ax.set_xlabel("Ending Balance (Nominal)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return fig, ax

def plot_percentile_bands(balances_ts: np.ndarray, ax=None, title="Balance Percentile Bands"):
    fig, ax = _ensure_ax(ax)
    p10 = np.percentile(balances_ts, 10, axis=0)
    p50 = np.percentile(balances_ts, 50, axis=0)
    p90 = np.percentile(balances_ts, 90, axis=0)
    t = np.arange(balances_ts.shape[1])
    ax.plot(t, p50, label="Median")
    ax.fill_between(t, p10, p90, alpha=0.3, label="10–90% band")
    ax.set_xlabel("Months")
    ax.set_ylabel("Balance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig, ax

def plot_survival_curve(failure_month: np.ndarray, T: int, ax=None, title="Survival Curve"):
    fig, ax = _ensure_ax(ax)
    n = failure_month.shape[0]
    surv = np.ones(T+1)
    for t in range(T+1):
        alive = ((failure_month == -1) | (failure_month >= t)).sum()
        surv[t] = alive / n
    ax.plot(np.arange(T+1), surv)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return fig, ax


def plot_simulated_annual_cashflows(
    nominal_median_by_year: np.ndarray,
    real_median_by_year: np.ndarray,
    ax=None,
    title_top: str = "Simulated Annual Cashflows (nominal)",
    title_bottom: str = "Simulated Annual Cashflows (in present dollars)",
):
    """
    Draws two stacked bar charts:
      - Top: median annual nominal cashflows (contribs+withdrawals) by year
      - Bottom: median annual real (present-dollar) cashflows by year
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    years = np.arange(1, len(nominal_median_by_year) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # --- Top: nominal ---
    axes[0].bar(years, nominal_median_by_year)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Median Annual Cashflow ($)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"${v:,.0f}"))
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].set_title(title_top)

    # --- Bottom: present dollars ---
    axes[1].bar(years, real_median_by_year)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Median Annual Cashflow ($)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"${v:,.0f}"))
    axes[1].grid(True, axis="y", alpha=0.2)
    axes[1].set_title(title_bottom)

    fig.tight_layout()
    return fig, axes



def plot_max_drawdown_histograms(
    mdd_including: np.ndarray,
    mdd_excluding: np.ndarray,
    bins: int = 20,
    clip_to_pct: int | None = 95,   # show middle 95% by default; set None to show all
    title_incl: str = "Maximum Drawdown Histogram Including Cashflows",
    title_excl: str = "Maximum Drawdown Histogram Excluding Cashflows",
):
    """
    Draw two histograms stacked vertically:
      - mdd_including: drawdowns computed on balance paths (includes cashflows)
      - mdd_excluding: drawdowns computed on pre-cashflow TWRR index (excludes cashflows)
    Inputs are decimals (e.g., -0.35 for -35%).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def _clip(data):
        if clip_to_pct is None:
            return data, None, None
        lo_q = (100 - clip_to_pct) / 2
        hi_q = 100 - lo_q
        lo, hi = np.percentile(data, [lo_q, hi_q])
        mask = (data >= lo) & (data <= hi)
        return data[mask], lo, hi

    inc, lo_i, hi_i = _clip(mdd_including)
    exc, lo_e, hi_e = _clip(mdd_excluding)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # Top: including cashflows
    axes[0].hist(inc, bins=bins, color="#ff2d2d")
    axes[0].set_title(
        f"{title_incl}" + (f" ({clip_to_pct}% of results)" if clip_to_pct else "")
    )
    axes[0].set_xlabel("Max. Drawdown")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1%}"))

    # Bottom: excluding cashflows
    axes[1].hist(exc, bins=bins, color="#ffa01e")
    axes[1].set_title(
        f"{title_excl}" + (f" ({clip_to_pct}% of results)" if clip_to_pct else "")
    )
    axes[1].set_xlabel("Max. Drawdown")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1%}"))

    fig.tight_layout()
    return fig, axes


def plot_correlation_matrix(
    returns_m,
    asset_labels=None,              # list of labels matching returns_m.columns (defaults to columns)
    title="Asset Correlation Matrix",
    subtitle=None,                  # e.g., "Based on monthly returns from Jan 1972 to Dec 2024"
    annotate=True                   # write correlation numbers inside cells
):
    import numpy as np
    import matplotlib.pyplot as plt

    cols = list(returns_m.columns)
    labels = asset_labels if asset_labels is not None else cols
    corr = returns_m.corr().values
    n = len(cols)

    # auto-size: wider for more assets
    fig_w = min(20, 5 + 0.8 * n)
    fig_h = min(20, 4 + 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=14)

    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Corr", rotation=270, va="bottom")

    if subtitle:
        fig.text(0.01, 0.01, subtitle, fontsize=10, color="#666666")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, ax


def plot_returns_risk_table(
    returns_m,
    asset_labels=None,              # list of labels matching returns_m.columns
    title="Asset Returns & Risk (Annualized)",
    subtitle=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from portfolio_tester.analytics.risk import asset_return_stats

    cols = list(returns_m.columns)
    labels = asset_labels if asset_labels is not None else cols

    stats = asset_return_stats(returns_m).loc[cols]  # keep same order
    # Format as percents for display
    stats_fmt = stats.applymap(lambda x: f"{x:.2%}")
    col_labels = ["CAGR", "Expected Annual Return", "Annualized Volatility"]
    cell_text = stats_fmt[col_labels].values.tolist()

    # Height scales with number of rows
    fig_w = 10
    fig_h = 2 + 0.45 * max(4, len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=labels,
        colLabels=col_labels,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.3)

    if subtitle:
        fig.text(0.01, 0.01, subtitle, fontsize=10, color="#666666")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, ax


def plot_survival_vs_vol(
    frontier,
    smoothed_rates: np.ndarray,
    best_idx: int,
    ax=None,
    title: str = "Survival Rate Across Constrained Frontier",
):
    """Simple helper to visualize survival vs. volatility for the frontier."""
    vols = np.array([fp.volatility for fp in frontier])
    raw = np.array([fp.survival_rate for fp in frontier])
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(vols, raw, marker="o", label="Raw survival")
    if smoothed_rates is not None:
        ax.plot(vols, smoothed_rates, linestyle="--", label="Smoothed survival")
    best = frontier[best_idx]
    ax.scatter([best.volatility], [best.survival_rate], s=180, marker="*", label="Best", zorder=5)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig, ax

def plot_constrained_frontier(
    frontier,
    best_idx: int,
    ax=None,
    title: str = "Constrained Efficient Frontier (Mean-Variance)",
):
    """Plot expected return vs volatility for the constrained frontier."""
    vols = np.array([fp.volatility for fp in frontier])
    rets = np.array([fp.expected_return for fp in frontier])
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(vols, rets, marker="o", label="Frontier")
    best = frontier[best_idx]
    ax.scatter([best.volatility], [best.expected_return], s=160, marker="*", label="Best survival", zorder=5)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Expected Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return fig, ax
