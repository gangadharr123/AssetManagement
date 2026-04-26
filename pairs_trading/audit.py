import pandas as pd
import numpy as np
import os, glob

print("=" * 70)
print("PAIRS TRADING PROJECT — FULL DATA & CODE AUDIT")
print("=" * 70)

# ── A. DATA FILES ────────────────────────────────────────────
print("\n[A] DATA FILES\n")

data_checks = {
    "Adjusted Prices": "outputs/data/sp500_adj_close.csv",
    "Volume": "outputs/data/sp500_volume.csv", 
    "Market Caps": "outputs/data/market_caps.csv",
    "Risk-Free Rate": "outputs/data/risk_free_daily.csv",
}

for name, path in data_checks.items():
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  ✅ {name}")
        print(f"     Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        if not df.empty:
            print(f"     Date range: {df.index[0]} to {df.index[-1]}")
        else:
            print(f"     Date range: EMPTY")
        if name == "Adjusted Prices":
            nan_pct = df.isna().mean().mean() * 100
            print(f"     NaN: {nan_pct:.1f}%")
            stocks_with_gaps = (df.isna().mean() > 0.2).sum()
            print(f"     Stocks with >20% missing: {stocks_with_gaps}")
            # Verify these are RAW adjusted prices (not already normalized to $1)
            sample = df.iloc[0][['AAPL','MSFT','GOOGL']].dropna()
            print(f"     First-day sample prices: {dict(sample.round(2))}")
            print(f"     (Should be real prices like 100-300, NOT $1.00)")
        if name == "Volume":
            zero_vol = (df == 0).any()
            print(f"     Stocks with any zero-vol day: {zero_vol.sum()}")
    else:
        print(f"  ❌ {name}: MISSING — {path}")

# ── B. REQUIRED DATE COVERAGE ────────────────────────────────
print("\n[B] DATE COVERAGE\n")
print("  Required: Jan 2015 through Dec 2024 minimum")
print("  First formation: Jan 2015 – Dec 2015")
print("  First trading:   Jan 2016 – Jun 2016") 
print("  Last formation:  ~Jul 2023 – Jun 2024")
print("  Last trading:    ~Jul 2024 – Dec 2024")
if os.path.exists("outputs/data/sp500_adj_close.csv"):
    p = pd.read_csv("outputs/data/sp500_adj_close.csv", index_col=0, parse_dates=True)
    start_ok = p.index[0].year <= 2015
    end_ok = p.index[-1].year >= 2024 and p.index[-1].month >= 12
    print(f"  Data starts ≤ 2015: {'✅' if start_ok else '❌'} ({p.index[0].date()})")
    print(f"  Data ends ≥ Dec 2024: {'✅' if end_ok else '❌'} ({p.index[-1].date()})")

# ── C. EXISTING OUTPUT FILES ─────────────────────────────────
print("\n[C] ALL FILES IN outputs/\n")
total_size = 0
for root, dirs, files in os.walk("outputs"):
    for f in sorted(files):
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        total_size += sz
        print(f"  {fp} ({sz/1024:.1f} KB)")
print(f"\n  Total: {total_size/1024/1024:.1f} MB")

# ── D. CODE AUDIT ────────────────────────────────────────────
print("\n[D] CODE FILES — METHODOLOGY CHECK\n")

code_files = {
    "config.py": "Configuration constants",
    "data/fetch_data.py": "Data download",
    "data/universe.py": "Universe filtering",
    "utils/windows.py": "Rolling window generator",
    "utils/transaction_costs.py": "Transaction cost model",
    "utils/portfolio_returns.py": "Monthly return aggregation",
    "strategies/pair_selection.py": "SSD computation",
    "strategies/distance_method.py": "Distance Method",
    "strategies/cointegration_method.py": "Cointegration Method",
    "strategies/copula_method.py": "Copula Method",
    "run_all.py": "Master runner",
}

for path, desc in code_files.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = len(f.readlines())
        print(f"  ✅ {path} ({lines} lines) — {desc}")
    else:
        print(f"  ❌ {path} — MISSING — {desc}")

# ── E. METHODOLOGY CROSS-CHECK ───────────────────────────────
print("\n[E] METHODOLOGY CROSS-CHECK WITH PAPER\n")

# Check config values
if os.path.exists("config.py"):
    with open("config.py", 'r') as f:
        config = f.read()
    checks = [
        ("FORMATION_MONTHS = 12", "12-month formation period"),
        ("TRADING_MONTHS = 6", "6-month trading period"),
        ("NUM_PAIRS = 20", "Top 20 pairs selected"),
        ("DM_THRESHOLD = 2", "2 std dev opening threshold for DM"),
        ("COINT_THRESHOLD = 2", "2 std dev for cointegration"),
        ("COPULA_THRESHOLD = 0.5", "0.5 threshold for copula M1/M2"),
    ]
    for pattern, desc in checks:
        key = pattern.split("=")[0].strip()
        found = key in config
        print(f"  {'✅' if found else '❌'} {desc}: {pattern}")

# Check DM implementation
if os.path.exists("strategies/distance_method.py"):
    with open("strategies/distance_method.py", 'r') as f:
        dm = f.read()
    print(f"\n  Distance Method checks:")
    print(f"    Normalizes to $1: {'✅' if '/ ' in dm or 'normalize' in dm.lower() else '⚠️  CHECK'}")
    print(f"    Uses 2 std threshold: {'✅' if '2' in dm and 'std' in dm.lower() else '⚠️  CHECK'}")
    print(f"    Closes at zero: {'✅' if '0' in dm and 'close' in dm.lower() else '⚠️  CHECK'}")
    print(f"    Re-normalizes at trading start: {'⚠️  MANUALLY VERIFY' }")

# Check cointegration
if os.path.exists("strategies/cointegration_method.py"):
    with open("strategies/cointegration_method.py", 'r') as f:
        ci = f.read()
    print(f"\n  Cointegration Method checks:")
    print(f"    Uses statsmodels coint: {'✅' if 'coint' in ci else '❌ MISSING'}")
    print(f"    Has beta/OLS: {'✅' if 'beta' in ci.lower() or 'OLS' in ci else '❌ MISSING'}")
    print(f"    Beta-weighted sizing: {'⚠️  MANUALLY VERIFY'}")

# Check copula
if os.path.exists("strategies/copula_method.py"):
    with open("strategies/copula_method.py", 'r') as f:
        co = f.read()
    print(f"\n  Copula Method checks:")
    print(f"    Fits marginals: {'✅' if 'logistic' in co.lower() or 'marginal' in co.lower() else '❌ MISSING'}")
    print(f"    Student-t copula: {'✅' if 'student' in co.lower() or 't_copula' in co.lower() else '❌ MISSING'}")
    print(f"    Cumulative M1/M2: {'✅' if 'M1' in co or 'cumul' in co.lower() else '❌ MISSING'}")
    print(f"    Threshold 0.5: {'✅' if '0.5' in co else '❌ MISSING'}")

print(f"\n{'='*70}")
print("AUDIT COMPLETE — Share this output for review before proceeding")
print(f"{'='*70}")
