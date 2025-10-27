import numpy as np
import pandas as pd
from ..config import SamplerConfig

class ReturnSampler:
    def __init__(self, rets_m: pd.DataFrame, infl_m: pd.Series):
        self.rets = rets_m.copy()
        self.infl = infl_m.reindex(rets_m.index).fillna(0.0)
        self.months = rets_m.index
        self.years = np.array([d.year for d in self.months])
        self.unique_years = np.unique(self.years)
        self.year_to_idx = {int(y): np.where(self.years == y)[0] for y in self.unique_years}

    def sample(self, horizon_m: int, n_sims: int, cfg: SamplerConfig):
        rng = np.random.default_rng(cfg.seed)
        A = self.rets.values  # (T, N)
        I = self.infl.values  # (T,)

        if cfg.mode == "single_month":
            idx = rng.integers(0, A.shape[0], size=(n_sims, horizon_m))
            R = A[idx, :]     # (n_sims, T, N)
            CPI = I[idx]      # (n_sims, T)
            return R, CPI

        elif cfg.mode == "single_year":
            blocks_needed = int(np.ceil(horizon_m / 12))
            year_choices = rng.choice(self.unique_years, size=(n_sims, blocks_needed))
            monthly_idx = []
            for s in range(n_sims):
                seq = []
                for y in year_choices[s]:
                    seq.extend(self.year_to_idx[int(y)].tolist())
                monthly_idx.append(seq[:horizon_m])
            monthly_idx = np.array(monthly_idx)
            R = A[monthly_idx, :]
            CPI = I[monthly_idx]
            return R, CPI

        elif cfg.mode == "block_years":
            k = int(cfg.block_years)
            ys = self.unique_years
            y_to_pos = {int(y): i for i, y in enumerate(ys)}
            blocks_needed = int(np.ceil(horizon_m / (12*k)))
            starts = rng.choice(ys, size=(n_sims, blocks_needed))
            monthly_idx = []
            for s in range(n_sims):
                idxs = []
                for start_y in starts[s]:
                    pos = y_to_pos[int(start_y)]
                    for j in range(k):
                        y = ys[(pos + j) % len(ys)]
                        idxs.extend(self.year_to_idx[int(y)].tolist())
                monthly_idx.append(idxs[:horizon_m])
            monthly_idx = np.array(monthly_idx)
            R = A[monthly_idx, :]
            CPI = I[monthly_idx]
            return R, CPI

        else:
            raise ValueError(f"Unknown sampling mode: {cfg.mode}")
