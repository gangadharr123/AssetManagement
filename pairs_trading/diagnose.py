import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from config import DATA_START, DATA_END, NUM_PAIRS, DM_THRESHOLD, COINT_THRESHOLD, COPULA_THRESHOLD, COINT_PVALUE, DATA_DIR, OUTPUT_DIR
from data.universe import filter_universe
from utils.windows import generate_windows
from strategies.pair_selection import compute_normalized_prices, compute_ssd, select_top_pairs
from strategies.distance_method import run_distance_method
from strategies.cointegration_method import find_cointegrated_pairs, run_cointegration_method
from strategies.copula_method import run_copula_method

def diagnose():
    prices = pd.read_csv(os.path.join(DATA_DIR, 'sp500_adj_close.csv'), index_col=0, parse_dates=True)
    volumes = pd.read_csv(os.path.join(DATA_DIR, 'sp500_volume.csv'), index_col=0, parse_dates=True)
    market_caps = pd.read_csv(os.path.join(DATA_DIR, 'market_caps.csv'), index_col=0)

    windows = generate_windows(prices)
    w = windows[0]
    
    fs, fe = w['formation_start'], w['formation_end']
    ts, te = w['trading_start'], w['trading_end']
    
    print(f"Window 0: Form {fs.date()} to {fe.date()}, Trade {ts.date()} to {te.date()}")
    
    universe = filter_universe(prices, volumes, market_caps, fs, fe)
    print(f"Universe size: {len(universe)}")
    
    norm_prices = compute_normalized_prices(prices, universe, fs, fe)
    ssd_ranked = compute_ssd(norm_prices)
    top_pairs = select_top_pairs(ssd_ranked, num_pairs=NUM_PAIRS)
    
    # DM
    dm_trades = run_distance_method(prices, top_pairs, fs, fe, ts, te, threshold=DM_THRESHOLD, window_id=0)
    print(f"\n--- DM TRADES (first 5 of {len(dm_trades)}) ---")
    for t in dm_trades[:5]:
        print(f"Pair: {t['pair']}, Entry: {t['entry_date'].date()}, Exit: {t['exit_date'].date()}, "
              f"Return: {t['return_before_cost']:.4f}, Dir: {t['direction']}, Converged: {t['converged']}")
              
    # Look closely at the math for the first DM trade
    if dm_trades:
        t = dm_trades[0]
        t1, t2 = t['pair']
        entry_p1 = prices.at[t['entry_date'], t1]
        entry_p2 = prices.at[t['entry_date'], t2]
        exit_p1 = prices.at[t['exit_date'], t1]
        exit_p2 = prices.at[t['exit_date'], t2]
        r1 = exit_p1 / entry_p1 - 1
        r2 = exit_p2 / entry_p2 - 1
        print("\nDM Math check for trade 1:")
        print(f"{t1}: Entry {entry_p1:.2f}, Exit {exit_p1:.2f}, Ret: {r1:.4f}")
        print(f"{t2}: Entry {entry_p2:.2f}, Exit {exit_p2:.2f}, Ret: {r2:.4f}")
        print(f"Computed trade return: {t['return_before_cost']:.4f}")
        
    # Cointegration
    print(f"\n--- COINTEGRATION ---")
    coint_pairs = find_cointegrated_pairs(prices, ssd_ranked, fs, fe, num_pairs=NUM_PAIRS, pvalue=0.05)
    print(f"Found {len(coint_pairs)} cointegrated pairs at p=0.05")
    if len(coint_pairs) == 0:
        coint_pairs_10 = find_cointegrated_pairs(prices, ssd_ranked, fs, fe, num_pairs=NUM_PAIRS, pvalue=0.10)
        print(f"Found {len(coint_pairs_10)} cointegrated pairs at p=0.10")
        
    # Copula
    print(f"\n--- COPULA ---")
    copula_trades = run_copula_method(prices, top_pairs, fs, fe, ts, te, threshold=COPULA_THRESHOLD, window_id=0)
    print(f"Copula trades: {len(copula_trades)}")

    # Save to outputs/tables
    os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
    if dm_trades:
        pd.DataFrame(dm_trades[:20]).to_csv(os.path.join(OUTPUT_DIR, 'tables', 'dm_sample_trades.csv'), index=False)
    if coint_pairs:
        # Generate trades for coint
        coint_trades = run_cointegration_method(prices, coint_pairs, ts, te, threshold=COINT_THRESHOLD, window_id=0)
        if coint_trades:
            pd.DataFrame(coint_trades[:20]).to_csv(os.path.join(OUTPUT_DIR, 'tables', 'coint_sample_trades.csv'), index=False)
    if copula_trades:
        pd.DataFrame(copula_trades[:20]).to_csv(os.path.join(OUTPUT_DIR, 'tables', 'copula_sample_trades.csv'), index=False)

if __name__ == '__main__':
    diagnose()
