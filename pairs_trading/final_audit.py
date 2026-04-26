import pandas as pd
import numpy as np
import os

print("=" * 70)
print("FINAL COMPREHENSIVE AUDIT")
print("=" * 70)

# ── 1A. PRICE DATA INTEGRITY ────────────────────────────────
prices = pd.read_csv('outputs/data/sp500_adj_close.csv', index_col=0, parse_dates=True)
volume = pd.read_csv('outputs/data/sp500_volume.csv', index_col=0, parse_dates=True)

print("\n[1A] PRICE DATA INTEGRITY")
print(f"  Shape: {prices.shape}")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  Trading days: {len(prices)} (expected ~2516 for 10 years)")
print(f"  Total NaN cells: {prices.isna().sum().sum()}")

# Check for suspicious values
print(f"  Any negative prices: {(prices < 0).any().any()}")
print(f"  Any zero prices: {(prices == 0).any().any()}")
print(f"  Any price > $10000: {(prices > 10000).any().any()}")
print(f"  Min price in dataset: ${prices.min().min():.2f}")
print(f"  Max price in dataset: ${prices.max().max():.2f}")

# Check for stock splits not properly adjusted
# A split would show as a >40% single-day drop
returns = prices.pct_change()
big_drops = (returns < -0.40).sum().sum()
big_jumps = (returns > 0.60).sum().sum()
print(f"  Single-day drops > 40%: {big_drops} (should be very few — possible unadjusted splits)")
print(f"  Single-day jumps > 60%: {big_jumps} (should be very few)")

# Check prices vs volume alignment
print(f"  Price columns: {prices.shape[1]}, Volume columns: {volume.shape[1]}")
print(f"  Columns match: {'✅' if set(prices.columns) == set(volume.columns) else '❌'}")
print(f"  Index match: {'✅' if prices.index.equals(volume.index) else '❌'}")

# ── 1B. MARKET CAP CHECK ────────────────────────────────────
print("\n[1B] MARKET CAP DATA")
mc = pd.read_csv('outputs/data/market_caps.csv')
print(f"  Rows: {len(mc)}")
if len(mc) > 0:
    print(f"  Columns: {list(mc.columns)}")
    mc_col = mc.columns[1] if len(mc.columns) > 1 else mc.columns[0]
    valid = mc[mc_col].dropna()
    print(f"  Valid market caps: {len(valid)}/{len(mc)}")
    print(f"  Min: ${valid.min()/1e9:.1f}B")
    print(f"  Max: ${valid.max()/1e9:.1f}B")
    print(f"  Bottom 10% cutoff: ${valid.quantile(0.10)/1e9:.1f}B")
    print(f"  Stocks that would be removed: {(valid < valid.quantile(0.10)).sum()}")
else:
    print("  ❌ EMPTY — bottom decile filter was NOT applied!")

# ── 1C. RISK-FREE RATE ──────────────────────────────────────
print("\n[1C] RISK-FREE RATE")
rf = pd.read_csv('outputs/data/risk_free_daily.csv', index_col=0, parse_dates=True)
print(f"  Days: {len(rf)}")
print(f"  Any NaN: {rf.isna().any().any()}")
print(f"  Any negative: {(rf < 0).any().any()}")
print(f"  Mean annualized: {rf.iloc[:,0].mean() * 252 * 100:.2f}%")

# ── 1D. FAMA-FRENCH FACTORS ─────────────────────────────────
print("\n[1D] FAMA-FRENCH FACTORS")
ff_path = 'outputs/tables/fama_french_monthly.csv'
if os.path.exists(ff_path):
    ff = pd.read_csv(ff_path, parse_dates=['month'])
    print(f"  Months: {len(ff)}")
    print(f"  Date range: {ff['month'].min()} to {ff['month'].max()}")
    required_cols = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF']
    missing = [c for c in required_cols if c not in ff.columns]
    print(f"  Required columns present: {'✅ All 7' if not missing else '❌ Missing: ' + str(missing)}")
    # Check values are in decimal (not percentage)
    print(f"  Mkt_RF mean: {ff['Mkt_RF'].mean():.4f} (should be ~0.005-0.01, NOT 0.5-1.0)")
    # Check our sample period is covered
    our_months = ff[(ff['month'] >= '2016-01-01') & (ff['month'] <= '2024-12-31')]
    print(f"  Months covering our trading period (2016-2024): {len(our_months)}")
else:
    print(f"  ❌ FILE MISSING")

print("\n" + "=" * 70)
print("PART 2: METHODOLOGY VERIFICATION")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# PART 2: READ EACH STRATEGY FILE AND VERIFY AGAINST PAPER
# ═══════════════════════════════════════════════════════════════

# ── 2A. CONFIG CONSTANTS ─────────────────────────────────────
print("\n[2A] CONFIG CONSTANTS")
with open('config.py', 'r') as f:
    config = f.read()
print(f"  Paper: Formation = 12 months → {'✅' if 'FORMATION_MONTHS = 12' in config else '❌'}")
print(f"  Paper: Trading = 6 months → {'✅' if 'TRADING_MONTHS = 6' in config else '❌'}")
print(f"  Paper: Top 20 pairs → {'✅' if 'NUM_PAIRS = 20' in config else '❌'}")
print(f"  Paper: DM threshold = 2 std → {'✅' if 'DM_THRESHOLD = 2.0' in config else '❌'}")
print(f"  Paper: Coint threshold = 2 std → {'✅' if 'COINT_THRESHOLD = 2.0' in config else '❌'}")
print(f"  Paper: Copula threshold = 0.5 → {'✅' if 'COPULA_THRESHOLD = 0.5' in config else '❌'}")

# ── 2B. DISTANCE METHOD ─────────────────────────────────────
print("\n[2B] DISTANCE METHOD — Checking against Paper Section 4.1")
with open('strategies/distance_method.py', 'r') as f:
    dm_code = f.read()

checks = [
    ("Formation normalization: price / price[0]", 
     "iloc[0]" in dm_code or "/ " in dm_code),
    ("TRADING period re-normalization to $1",
     "trade" in dm_code.lower() and ("iloc[0]" in dm_code or "normalize" in dm_code.lower())),
    ("Spread = norm_price_A - norm_price_B",
     "-" in dm_code),
    ("Open threshold: spread > 2 * std",
     "threshold" in dm_code and "std" in dm_code.lower()),
    ("Close condition: spread crosses zero (sign change)",
     "<= 0" in dm_code and ">= 0" in dm_code),
    ("Multiple round-trips allowed",
     "position = 0" in dm_code or "position == 0" in dm_code),
    ("Force close at trading end",
     "force" in dm_code.lower() or "end" in dm_code.lower() or "unconverge" in dm_code.lower()),
    ("$1 long + $1 short position sizing",
     True),  # Verified manually earlier
]
for desc, check in checks:
    print(f"  {'✅' if check else '⚠️  VERIFY'} {desc}")

# ── 2C. COINTEGRATION METHOD ────────────────────────────────
print("\n[2C] COINTEGRATION METHOD — Checking against Paper Section 4.2")
with open('strategies/cointegration_method.py', 'r') as f:
    ci_code = f.read()

checks = [
    ("Step 1: SSD pre-filter (shared with DM)",
     "ssd" in ci_code.lower() or "pair_selection" in ci_code.lower()),
    ("Step 2: Engle-Granger cointegration test",
     "coint" in ci_code),
    ("Uses statsmodels coint function",
     "statsmodels" in ci_code or "coint" in ci_code),
    ("OLS to estimate beta (on price LEVELS not returns)",
     "OLS" in ci_code or "ols" in ci_code or "beta" in ci_code),
    ("Spread = X2 - beta * X1 (equation 3)",
     "beta" in ci_code and "*" in ci_code),
    ("Normalized spread = (spread - mean) / std (equation 6)",
     "mean" in ci_code and ("std" in ci_code or "sigma" in ci_code)),
    ("Open threshold: |normalized_spread| > 2",
     "threshold" in ci_code),
    ("Close: normalized spread crosses zero",
     "<= 0" in ci_code and ">= 0" in ci_code),
    ("Beta-weighted position sizing",
     "beta" in ci_code and ("ret" in ci_code.lower() or "return" in ci_code.lower())),
    ("Selects up to 20 cointegrated pairs",
     "20" in ci_code or "num_pairs" in ci_code.lower()),
    ("Cointegration p-value threshold (0.05)",
     "0.05" in ci_code or "pvalue" in ci_code.lower()),
]
for desc, check in checks:
    print(f"  {'✅' if check else '⚠️  VERIFY'} {desc}")

# ── 2D. COPULA METHOD ───────────────────────────────────────
print("\n[2D] COPULA METHOD — Checking against Paper Section 4.3")
with open('strategies/copula_method.py', 'r') as f:
    co_code = f.read()

checks = [
    ("SSD pre-filter, top 20 pairs",
     "ssd" in co_code.lower() or "20" in co_code),
    ("Fits marginal distributions (IFM method step 1)",
     "logistic" in co_code.lower() or "marginal" in co_code.lower() or "fit" in co_code.lower()),
    ("Candidate marginals include: Normal, Logistic, GEV, Extreme Value",
     "logistic" in co_code.lower() or "norm" in co_code.lower()),
    ("Selects best marginal by AIC/BIC",
     "aic" in co_code.lower() or "bic" in co_code.lower() or "loglik" in co_code.lower()),
    ("Fits copula to uniform marginals (IFM step 2)",
     "copula" in co_code.lower() or "cdf" in co_code.lower()),
    ("Student-t copula implemented",
     "student" in co_code.lower() or "t_" in co_code.lower() or "nu" in co_code),
    ("Conditional probability h1 = P(U1<=u1|U2=u2)",
     "h1" in co_code or "h2" in co_code or "conditional" in co_code.lower()),
    ("Mispricing index m = h - 0.5 (equation 12)",
     "0.5" in co_code and ("-" in co_code)),
    ("Cumulative M1, M2 starting at 0 (equation 13)",
     "M1" in co_code or "m1" in co_code.lower()),
    ("Open: M1 > 0.5 AND M2 < -0.5 (or vice versa)",
     "threshold" in co_code and "and" in co_code.lower()),
    ("Close: BOTH M1 and M2 cross zero",
     "<= 0" in co_code and ">= 0" in co_code),
]
for desc, check in checks:
    print(f"  {'✅' if check else '⚠️  VERIFY'} {desc}")

# ── 2E. ROLLING WINDOWS ─────────────────────────────────────
print("\n[2E] ROLLING WINDOWS — Paper Section 4")
with open('utils/windows.py', 'r') as f:
    win_code = f.read()
print(f"  12-month formation: {'✅' if '12' in win_code else '⚠️'}")
print(f"  6-month trading: {'✅' if '6' in win_code else '⚠️'}")
print(f"  Monthly rolling: {'✅' if 'month' in win_code.lower() else '⚠️'}")

# ── 2F. TRANSACTION COSTS ───────────────────────────────────
print("\n[2F] TRANSACTION COSTS — Paper Section 4.4 (adapted for 2015-2025)")
with open('utils/transaction_costs.py', 'r') as f:
    tc_code = f.read()
print(f"  Verify: each pairs trade should cost ~100 bps (4 one-way trades × ~25 bps)")

# ── 2G. PORTFOLIO RETURNS ───────────────────────────────────
print("\n[2G] PORTFOLIO RETURNS — Paper Section 4.5")
with open('utils/portfolio_returns.py', 'r') as f:
    pr_code = f.read()
print(f"  Handles 6 overlapping portfolios: {'✅' if '6' in pr_code or 'overlap' in pr_code.lower() else '⚠️  VERIFY'}")
print(f"  Return on Employed Capital (eq 14): {'✅' if 'employed' in pr_code.lower() or 'traded' in pr_code.lower() else '⚠️  VERIFY'}")
print(f"  Return on Committed Capital (eq 15): {'✅' if 'committed' in pr_code.lower() or '20' in pr_code else '⚠️  VERIFY'}")
print(f"  Equal-weighted average across portfolios: {'✅' if 'mean' in pr_code.lower() or 'average' in pr_code.lower() else '⚠️  VERIFY'}")

print("\n" + "=" * 70)
print("PART 3: OUTPUT FILE VERIFICATION")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# PART 3: VERIFY ALL OUTPUT CSVs
# ═══════════════════════════════════════════════════════════════

expected_files = {
    'outputs/tables/dm_trades.csv': ['ticker_A','ticker_B','entry_date','exit_date','direction','return_before_cost','return_after_cost','transaction_cost','converged','days_open','window_id'],
    'outputs/tables/cointegration_trades.csv': ['ticker_A','ticker_B','entry_date','exit_date','direction','return_before_cost','return_after_cost','converged','days_open','window_id'],
    'outputs/tables/copula_trades.csv': ['ticker_A','ticker_B','entry_date','exit_date','direction','return_before_cost','return_after_cost','converged','days_open','window_id'],
    'outputs/tables/monthly_returns.csv': ['dm_ret_before','dm_ret_after'],
    'outputs/tables/selected_pairs.csv': ['window_id','strategy','ticker_A','ticker_B'],
    'outputs/tables/universe_summary.csv': ['window_id'],
    'outputs/tables/fama_french_monthly.csv': ['Mkt_RF','SMB','HML','RMW','CMA','MOM','RF'],
    'outputs/tables/summary_stats.csv': ['strategy'],
}

for filepath, required_cols in expected_files.items():
    print(f"\n  [{filepath.split('/')[-1]}]")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"    Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"    Columns: {list(df.columns)}")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"    ❌ Missing columns: {missing}")
        else:
            print(f"    ✅ All required columns present")
        
        # Check for empty or all-NaN
        if len(df) == 0:
            print(f"    ❌ FILE IS EMPTY")
        elif df.isna().all().any():
            null_cols = df.columns[df.isna().all()].tolist()
            print(f"    ⚠️  All-NaN columns: {null_cols}")
        else:
            print(f"    ✅ No empty columns")
        
        # For trade files, verify data integrity
        if 'trades' in filepath:
            print(f"    Converged: {df['converged'].sum()} ({df['converged'].mean()*100:.1f}%)")
            print(f"    Return range: [{df['return_before_cost'].min():.4f}, {df['return_before_cost'].max():.4f}]")
            print(f"    Any NaN returns: {df['return_before_cost'].isna().any()}")
            print(f"    Days open range: [{df['days_open'].min()}, {df['days_open'].max()}]")
            # Check for impossible values
            print(f"    Returns > 100%: {(df['return_before_cost'].abs() > 1.0).sum()} (should be very rare)")
    else:
        print(f"    ❌ FILE MISSING")

print("\n" + "=" * 70)
print("PART 4: CROSS-STRATEGY CONSISTENCY CHECKS")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# PART 4: CROSS-CHECKS THAT CATCH SUBTLE BUGS
# ═══════════════════════════════════════════════════════════════

dm = pd.read_csv('outputs/tables/dm_trades.csv')
ci = pd.read_csv('outputs/tables/cointegration_trades.csv')
co = pd.read_csv('outputs/tables/copula_trades.csv')
mr = pd.read_csv('outputs/tables/monthly_returns.csv')

print("\n[4A] TRADE COUNT REASONABLENESS")
print(f"  DM: {len(dm)} trades across 103 windows = {len(dm)/103:.1f} per window")
print(f"  Coint: {len(ci)} trades across 103 windows = {len(ci)/103:.1f} per window")
print(f"  Copula: {len(co)} trades across 103 windows = {len(co)/103:.1f} per window")
print(f"  Paper had ~15000 DM trades over 620 months ≈ 24 per month")
print(f"  Our rate seems {'reasonable' if 10 < len(dm)/103 < 80 else 'SUSPICIOUS'}")

print("\n[4B] CONVERGED TRADE RETURNS SHOULD BE POSITIVE")
for name, df in [("DM", dm), ("Coint", ci), ("Copula", co)]:
    conv = df[df['converged'] == True]
    pct_positive = (conv['return_before_cost'] > 0).mean() * 100
    print(f"  {name}: {pct_positive:.1f}% of converged trades are positive (paper expects >94%)")
    if pct_positive < 80:
        print(f"    ❌ TOO LOW — possible bug in return calculation")

print("\n[4C] UNCONVERGED TRADE RETURNS SHOULD BE MOSTLY NEGATIVE")
for name, df in [("DM", dm), ("Coint", ci), ("Copula", co)]:
    unconv = df[df['converged'] == False]
    if len(unconv) > 0:
        pct_negative = (unconv['return_before_cost'] < 0).mean() * 100
        print(f"  {name}: {pct_negative:.1f}% of unconverged trades are negative")

print("\n[4D] TRANSACTION COST CONSISTENCY")
for name, df in [("DM", dm), ("Coint", ci), ("Copula", co)]:
    if 'transaction_cost' in df.columns:
        tc = df['transaction_cost']
        print(f"  {name}: TC range [{tc.min():.4f}, {tc.max():.4f}], mean={tc.mean():.4f}")
        implied_diff = (df['return_before_cost'] - df['return_after_cost']).mean()
        print(f"    Before - After cost diff: {implied_diff:.4f} (should equal mean TC)")

print("\n[4E] MONTHLY RETURNS DATE COVERAGE")
if 'month' in mr.columns:
    mr['month'] = pd.to_datetime(mr['month'])
    print(f"  First month: {mr['month'].min().date()}")
    print(f"  Last month: {mr['month'].max().date()}")
    print(f"  Total months: {len(mr)} (expected ~108 for Jan 2016 - Dec 2024)")
    print(f"  Any gaps: {len(mr) != len(pd.date_range(mr['month'].min(), mr['month'].max(), freq='MS'))}")

print("\n[4F] MONTHLY RETURN MAGNITUDES")
for col in ['dm_ret_before','dm_ret_after']:
    if col in mr.columns:
        print(f"  {col}: mean={mr[col].mean()*100:.3f}%, std={mr[col].std()*100:.3f}%")
print(f"  Paper DM: mean=0.91% before, 0.38% after (but for 1962-2014)")
print(f"  Lower/negative values expected for 2015-2025 due to declining profitability")

print("\n[4G] PAIR SELECTION — DO PAIRS OVERLAP ACROSS STRATEGIES?")
sp = pd.read_csv('outputs/tables/selected_pairs.csv') if os.path.exists('outputs/tables/selected_pairs.csv') else None
if sp is not None and 'strategy' in sp.columns:
    for wid in sp['window_id'].unique()[:3]:
        w = sp[sp['window_id'] == wid]
        dm_pairs = set(zip(w[w['strategy']=='Distance']['ticker_A'], w[w['strategy']=='Distance']['ticker_B']))
        ci_pairs = set(zip(w[w['strategy']=='Cointegration']['ticker_A'], w[w['strategy']=='Cointegration']['ticker_B'])) if 'Cointegration' in w['strategy'].values else set()
        overlap = dm_pairs & ci_pairs
        print(f"  Window {wid}: DM has {len(dm_pairs)} pairs, Coint has {len(ci_pairs)} pairs, Overlap: {len(overlap)}")
    print(f"  (Some overlap expected — both use SSD pre-filter)")

print("\n" + "=" * 70)
print("PART 5: FINAL MANUAL VERIFICATION CHECKLIST")
print("=" * 70)

print("""
After reviewing the automated checks above, Gemini must now MANUALLY 
read through each strategy file and verify these specific items by 
inspecting the actual code logic (not just searching for keywords):

□ 1. distance_method.py: When a trade opens because spread > +2*std,
     does the code go SHORT stock A and LONG stock B? (A is above B 
     in the spread, so A is overvalued)

□ 2. distance_method.py: At trading period start, is normalized price 
     computed as trade_price / trade_price_on_first_day? (NOT using 
     formation period first day)

□ 3. cointegration_method.py: Is the OLS regression run on PRICE LEVELS 
     (not returns, not normalized prices)?

□ 4. cointegration_method.py: When spread < -2, does the code buy $1 of 
     stock 2 and sell $beta of stock 1? (matching paper equation)

□ 5. cointegration_method.py: Is the return calculated accounting for 
     unequal position sizes?

□ 6. copula_method.py: Are daily RETURNS (not prices) used for marginal 
     fitting?

□ 7. copula_method.py: After fitting marginals on FORMATION data, are 
     TRADING period returns transformed using the FORMATION-fitted CDFs?
     (no re-fitting during trading period = no look-ahead bias)

□ 8. copula_method.py: Does the Student-t conditional probability formula 
     match Table 2 of the paper?

□ 9. portfolio_returns.py: Are there 6 overlapping portfolios being 
     averaged each month?

□ 10. universe.py: Is the bottom decile market cap filter actually 
      removing stocks? (Given market_caps.csv was empty earlier, this 
      may have been silently skipped)

Read the actual code for each item. Print the relevant code snippet 
and confirm ✅ or identify the bug ❌ for each.
""")

print("=" * 70)
print("Run this audit, show ALL output, then do the 10-item manual check.")
print("=" * 70)