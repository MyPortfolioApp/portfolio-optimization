from dataclasses import dataclass
from typing import List, Literal, Optional

@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    weight: float  # 0..1

@dataclass(frozen=True)
class Portfolio:
    assets: List[Asset]

    def weights_vector(self):
        import numpy as np
        return np.array([a.weight for a in self.assets], dtype=float)

    def tickers(self):
        return [a.ticker for a in self.assets]

@dataclass(frozen=True)
class Goal:
    name: str
    amount: float           # +contribution, -withdrawal (end of month)
    start_month: int        # 0=now, 12=in 1 year
    frequency: Literal[1,4,12]  # times per year: 1=annual, 4=quarterly, 12=monthly
    repeats: int            # number of payments
    real: bool = False      # True => index by sampled CPI to the payment date

@dataclass(frozen=True)
class DataConfig:
    start: Optional[str] = None    # 'YYYY-MM-01'
    end: Optional[str] = None
    force_common_overlap: bool = True

@dataclass(frozen=True)
class SamplerConfig:
    mode: Literal["single_month", "single_year", "block_years"] = "single_year"
    block_years: int = 5
    seed: Optional[int] = 42

@dataclass(frozen=True)
class SimConfig:
    horizon_months: int
    n_sims: int = 10_000
    rebalance_every_months: int = 12
    starting_balance: float = 1_000_000.0
