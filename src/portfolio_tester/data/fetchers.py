import pandas as pd
import numpy as np
from .cache import key_path
import yfinance as yf

def _cache_read(path):
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None

def _cache_write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def log(msg: str):
    print(f"[debug] {msg}")

def fetch_prices_monthly(tickers, start=None, end=None):
    """Download daily auto-adjusted prices from Yahoo and resample to month-end."""

    if isinstance(tickers, str):
        tickers = [tickers]

    key = f"{','.join(tickers)}|{start}|{end}"
    path = key_path("prices", key)
    cached = _cache_read(path)
    if cached is not None:
        return cached

    log(f"Downloading from Yahoo Finance: {tickers}")
    download_kwargs = dict(
        tickers=tickers,
        auto_adjust=True,
        progress=False,
        interval="1d",
        group_by="column",
    )
    if start is not None:
        download_kwargs["start"] = pd.to_datetime(start)
    if end is not None:
        download_kwargs["end"] = pd.to_datetime(end)
    if start is None and end is None:
        download_kwargs["period"] = "max"
    data = yf.download(**download_kwargs)

    def extract_close_frame(data, tickers):
        import pandas as pd
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            lvl0 = set(data.columns.get_level_values(0))
            if "Close" in lvl0:
                return data["Close"].copy()
            if "Adj Close" in lvl0:
                return data["Adj Close"].copy()
            for fld in ("Close", "Adj Close"):
                try:
                    return data.xs(fld, level=1, axis=1).copy()
                except KeyError:
                    pass
        if isinstance(data, pd.DataFrame):
            for fld in ("Close", "Adj Close"):
                if fld in data.columns:
                    return data[[fld]].rename(columns={fld: tickers[0]}).copy()
        raise RuntimeError(f"Could not find Close/Adj Close columns. Columns={data.columns}")

    px_daily = extract_close_frame(data, tickers)
    present = [t for t in tickers if t in px_daily.columns]
    if not present:
        raise RuntimeError("None of the requested tickers returned price data.")
    px_daily = px_daily[present]
    monthly = px_daily.resample("ME").last().dropna(how="all")
    _cache_write(monthly, path)
    return monthly
 

def fetch_fred_series(series_id, start=None, end=None):
    """
    Fetch a FRED series via CSV endpoints (no API key), bypassing system proxies.
    Tries 'downloaddata' then 'fredgraph' CSV. Caches monthly, month-end data.
    """
    import io
    import pandas as pd
    import requests, certifi

    key = f"{series_id}|{start}|{end}"
    path = key_path("fred", key)
    cached = _cache_read(path)
    if cached is not None:
        return cached

    urls = [
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv&frequency=m",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&frequency=m",
    ]

    sess = requests.Session()
    # CRITICAL: ignore proxy environment variables that may be misconfigured
    sess.trust_env = False
    headers = {"User-Agent": "portfolio-tester/0.1"}

    last_exc = None
    for url in urls:
        try:
            r = sess.get(
                url,
                timeout=30,
                verify=certifi.where(),
                headers=headers,
                allow_redirects=True,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))

            # Normalize CSV: expect a observation_date column + value column
            if "observation_date" not in df.columns:
                raise ValueError("CSV missing observation_date column")
            df["observation_date"] = pd.to_datetime(df["observation_date"])

            # Pick the value column
            if series_id in df.columns:
                value_col = series_id
            else:
                # fredgraph.csv returns the series as a non-observation_date column
                value_cols = [c for c in df.columns if c != "observation_date"]
                if not value_cols:
                    raise ValueError("CSV missing value column")
                value_col = value_cols[0]
                df = df.rename(columns={value_col: series_id})

            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            df = df.dropna(subset=[series_id]).set_index("observation_date")[[series_id]]

            if start is not None:
                df = df[df.index >= pd.to_datetime(start)]
            if end is not None:
                df = df[df.index <= pd.to_datetime(end)]

            df = df.resample("ME").last()
            _cache_write(df, path)
            return df
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_exc}")

def prep_returns_and_macro(prices_m, start=None, end=None):
    """
    - Determine effective start from first fully populated price row + optional override
    - Fill interior gaps with neighbor averages to keep uninterrupted history
    - Compute returns and macro series aligned to the trimmed window
    Returns: (returns_df, inflation_series, riskfree_series)
    """

    def _fill_neighbor_average(df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()
        for column in filled.columns:
            series = filled[column]
            missing_idx = series[series.isna()].index
            for idx in missing_idx:
                loc = series.index.get_loc(idx)
                prev_vals = series.iloc[:loc].dropna()
                next_vals = series.iloc[loc + 1 :].dropna()
                if not prev_vals.empty and not next_vals.empty:
                    filled.at[idx, column] = (prev_vals.iloc[-1] + next_vals.iloc[0]) / 2.0
                elif not prev_vals.empty:
                    filled.at[idx, column] = prev_vals.iloc[-1]
                elif not next_vals.empty:
                    filled.at[idx, column] = next_vals.iloc[0]
        return filled

    prices_m = prices_m.sort_index().dropna(axis=1, how="all")

    complete_rows = prices_m.dropna(how="any")
    if complete_rows.empty:
        raise ValueError("No overlapping price data available across tickers.")
    feasible_start = complete_rows.index.min()

    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None
    effective_start = feasible_start if start_ts is None else max(feasible_start, start_ts)

    if end_ts is not None and end_ts < effective_start:
        raise ValueError("end precedes the earliest feasible start date.")

    price_window = prices_m.loc[prices_m.index >= effective_start]
    if end_ts is not None:
        price_window = price_window.loc[price_window.index <= end_ts]

    price_window = _fill_neighbor_average(price_window).dropna(how="any")
    if price_window.shape[0] < 2:
        raise ValueError("Insufficient price history after gap filling to compute returns.")

    rets_m = price_window.pct_change().dropna()
    if rets_m.empty:
        raise ValueError("Insufficient observations to compute monthly returns.")

    macro_start, macro_end = rets_m.index.min(), rets_m.index.max()

    cpi = fetch_fred_series("CPIAUCSL", start=macro_start, end=macro_end).reindex(rets_m.index)
    cpi = _fill_neighbor_average(cpi)
    infl_m = cpi["CPIAUCSL"].pct_change(fill_method=None).fillna(0.0)

    tb3 = fetch_fred_series("TB3MS", start=macro_start, end=macro_end).reindex(rets_m.index)
    tb3 = _fill_neighbor_average(tb3)
    rf_m = ((1.0 + (tb3["TB3MS"] / 100.0)) ** (1 / 12.0) - 1.0).ffill().fillna(0.0)

    return rets_m, infl_m, rf_m

