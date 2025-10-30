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

        def trim_to_full_years():
            full_years = [int(y) for y, idx in self.year_to_idx.items() if len(idx) == 12]
            if not full_years:
                raise ValueError("No full 12-month years available for year-based sampling.")
            idx_arrays = [self.year_to_idx[y] for y in full_years]
            concat_idx = np.concatenate(idx_arrays) if len(idx_arrays) > 1 else idx_arrays[0]
            months = self.months[concat_idx]
            trimmed_years = np.array([d.year for d in months])
            trimmed_year_to_idx = {int(y): np.where(trimmed_years == y)[0] for y in full_years}
            return (
                self.rets.values[concat_idx, :],
                self.infl.values[concat_idx],
                trimmed_years,
                np.array(full_years, dtype=int),
                trimmed_year_to_idx,
            )

        if cfg.mode == "single_month":
            idx = rng.integers(0, A.shape[0], size=(n_sims, horizon_m))
            R = A[idx, :]     # (n_sims, T, N)
            CPI = I[idx]      # (n_sims, T)
            return R, CPI

        elif cfg.mode == "single_year":
            A, I, _, unique_years, year_to_idx = trim_to_full_years()
            blocks_needed = int(np.ceil(horizon_m / 12))
            year_choices = rng.choice(unique_years, size=(n_sims, blocks_needed))
            monthly_idx = []
            for s in range(n_sims):
                seq = []
                for y in year_choices[s]:
                    seq.extend(year_to_idx[int(y)].tolist())
                monthly_idx.append(seq[:horizon_m])
            monthly_idx = np.array(monthly_idx)
            R = A[monthly_idx, :]
            CPI = I[monthly_idx]
            return R, CPI

        elif cfg.mode == "block_years":
            A, I, _, unique_years, year_to_idx = trim_to_full_years()
            k = int(cfg.block_years)
            ys = unique_years
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
                        idxs.extend(year_to_idx[int(y)].tolist())
                monthly_idx.append(idxs[:horizon_m])
            monthly_idx = np.array(monthly_idx)
            R = A[monthly_idx, :]
            CPI = I[monthly_idx]
            return R, CPI

        else:
            raise ValueError(f"Unknown sampling mode: {cfg.mode}")
