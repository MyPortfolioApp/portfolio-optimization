# scripts/fetch_debug.py
# Run:  python scripts/fetch_debug.py

import io
import os
import sys
import textwrap
from datetime import datetime

import pandas as pd

# --- Optional: hard-disable proxy env vars for this run (uncomment if needed) ---
# for k in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy","NO_PROXY","no_proxy"]:
#     os.environ.pop(k, None)

def log(msg: str):
    print(f"[fetch_debug] {msg}")

def fetch_fred_csv(series_id: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Fetch a FRED series using CSV endpoints (no API key). We:
    - ignore proxy env vars
    - try 2 official CSV forms
    - verify content-type is CSV (or at least plain text)
    - tolerate different date/value column names
    Returns a DataFrame indexed by date with a single column = series_id.
    """
    import requests, certifi

    urls = [
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        # last-resort CSV API (usually works w/o key, but not guaranteed):
        f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&file_type=csv",
    ]

    sess = requests.Session()
    sess.trust_env = False  # ignore system proxies
    headers = {"User-Agent": "portfolio-tester-debug/0.1"}

    last_exc = None
    for url in urls:
        try:
            log(f"GET {url}")
            r = sess.get(
                url,
                timeout=30,
                verify=certifi.where(),
                headers=headers,
                allow_redirects=True,
                proxies={"http": None, "https": None},
            )
            ctype = r.headers.get("Content-Type", "")
            log(f"  -> HTTP {r.status_code} Content-Type={ctype!r}")
            r.raise_for_status()

            # We expect text/csv (or text/plain). If HTML, show first chars for diagnosis.
            if "csv" not in ctype.lower() and "text" not in ctype.lower():
                preview = r.text[:200].replace("\n", "\\n")
                raise RuntimeError(f"Unexpected content type: {ctype}; preview='{preview}...'")

            df = pd.read_csv(io.StringIO(r.text))
            cols_lower = {c.lower(): c for c in df.columns}

            # FRED CSV variants: DATE or observation_date
            date_col = cols_lower.get("date") or cols_lower.get("observation_date")
            if date_col is None:
                # dump first few lines to help debugging
                preview = r.text.splitlines()[:5]
                raise RuntimeError("CSV missing a DATE/observation_date column. First lines: " + " | ".join(preview))

            # Identify the value column
            if series_id in df.columns:
                val_col = series_id
            else:
                # assume the first non-date column
                non_date = [c for c in df.columns if c != date_col]
                if not non_date:
                    raise RuntimeError("CSV missing value column.")
                val_col = non_date[0]
                df = df.rename(columns={val_col: series_id})

            # Clean and filter
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            df = df.dropna(subset=[series_id])

            if start:
                df = df[df.index >= pd.to_datetime(start)]
            if end:
                df = df[df.index <= pd.to_datetime(end)]

            # Convert to month-end frequency
            df = df.resample("ME").last()
            return df[[series_id]]

        except Exception as e:
            last_exc = e
            log(f"  !! Attempt failed: {e}")

    raise RuntimeError(f"All FRED CSV attempts failed for {series_id}: {last_exc}")

def fetch_yahoo_prices_monthly(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily auto-adjusted prices (dividends/splits applied) from Yahoo via yfinance,
    extract the Close series, and resample to month end.
    """
    import yfinance as yf

    log(f"Downloading from Yahoo Finance: {tickers}")
    data = yf.download(
        tickers,
        auto_adjust=True,  # gives 'Close' with dividends/splits applied
        progress=False,
        interval="1d",
        group_by="column",
        period="max",  # longest available
    )

    # Normalize to a simple DataFrame with columns=tickers
    def extract_close_frame(data, tickers):
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            lvl0 = set(data.columns.get_level_values(0))
            if "Close" in lvl0:
                return data["Close"].copy()
            if "Adj Close" in lvl0:
                return data["Adj Close"].copy()
            # ticker-first layout (rare)
            for fld in ("Close", "Adj Close"):
                try:
                    return data.xs(fld, level=1, axis=1).copy()
                except KeyError:
                    pass
        if isinstance(data, pd.DataFrame):
            for fld in ("Close", "Adj Close"):
                if fld in data.columns:
                    return data[[fld]].rename(columns={fld: tickers[0]}).copy()
        # If we got here, diagnostic dump:
        raise RuntimeError(f"Could not find Close/Adj Close columns. Columns={data.columns}")

    px_daily = extract_close_frame(data, tickers)
    present = [t for t in tickers if t in px_daily.columns]
    if not present:
        raise RuntimeError("None of the requested tickers returned price data.")
    px_daily = px_daily[present]
    monthly = px_daily.resample("ME").last().dropna(how="all")
    return monthly

def main():
    # --- Change these if you want ---
    tickers = ["VTI", "TLT", "IEF", "GSG", "GLD"]
    fred_series = ["CPIAUCSL", "TB3MS"]

    # 1) Yahoo monthly prices
    try:
        px_m = fetch_yahoo_prices_monthly(tickers)
        print("\n=== Yahoo monthly prices (head) ===")
        print(px_m.head(10))
        print("\n=== Yahoo monthly prices (tail) ===")
        print(px_m.tail(10))
        print(f"\nYahoo shape: {px_m.shape} (rows x cols)\n")
        px_m.to_csv("yahoo_prices_monthly.csv")
        log("Saved: yahoo_prices_monthly.csv")
    except Exception as e:
        log(f"YAHOO ERROR: {e}")

    # 2) FRED series (monthly)
    for sid in fred_series:
        try:
            df = fetch_fred_csv(sid)
            print(f"\n=== FRED {sid} monthly (head) ===")
            print(df.head(12))
            print(f"\n=== FRED {sid} monthly (tail) ===")
            print(df.tail(12))
            print(f"\nFRED {sid} shape: {df.shape}\n")
            df.to_csv(f"fred_{sid}_monthly.csv")
            log(f"Saved: fred_{sid}_monthly.csv")
        except Exception as e:
            log(f"FRED ERROR for {sid}: {e}")

if __name__ == "__main__":
    main()
