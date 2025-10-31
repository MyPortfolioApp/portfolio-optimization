import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple

def _ensure_ax(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    else:
        fig = ax.figure
    return fig, ax

def plot_allocation_donut(labels: Sequence[str], weights: Sequence[float], ax=None, title="Portfolio Allocation"):
    """Donut chart showing allocation percentages on wedges and in the legend."""
    import numpy as np

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
