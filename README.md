# Portfolio Tester (MVP)

TEST CIAO

This is a minimal, runnable scaffold for the Monte Carlo portfolio tester.

## How to run in VS Code

### 1) Open the folder
- Download and unzip this project locally.
- In VS Code, File → Open Folder… and select the unzipped folder.

### 2) Create a virtual environment
**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

> Tip: If you have a FRED API key, set `FRED_API_KEY` in your environment.
> The code also falls back to `pandas-datareader` if no key is set.

### 4) Run the quickstart (100 simulations)
```bash
python scripts/quickstart.py
```

You should see a short summary (survival %, median end balance, CAGR, etc.).
Then you can tweak inputs in `scripts/quickstart.py`:
- Portfolio tickers & weights
- Horizon (months), sampler mode
- Cashflow goals (amounts, start offset, frequency, repeats, real vs nominal)

Bellaaa