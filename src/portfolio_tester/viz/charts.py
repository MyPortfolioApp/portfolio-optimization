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
    fig, ax = _ensure_ax(ax)
    wedges, _ = ax.pie(weights, startangle=90, wedgeprops=dict(width=0.4))
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    return fig, ax

def plot_efficient_frontier(risks: np.ndarray, rets: np.ndarray, port_pt: Tuple[float,float]=None, ax=None, title="Efficient Frontier"):
    fig, ax = _ensure_ax(ax)
    ax.plot(risks, rets, lw=2, label="Frontier")
    if port_pt is not None:
        ax.scatter([port_pt[0]], [port_pt[1]], s=60, marker="o", label="Current")
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
