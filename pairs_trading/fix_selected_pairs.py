import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

from config import DATA_DIR, OUTPUT_DIR, NUM_PAIRS, COINT_PVALUE
from data.universe import filter_universe
from utils.windows import generate_windows
from strategies.pair_selection import compute_normalized_prices, compute_ssd, select_top_pairs
from strategies.cointegration_method import find_cointegrated_pairs

def fix_selected_pairs():
    print("Reconstructing selected_pairs.csv for all strategies...")
    
    prices = pd.read_csv(os.path.join(DATA_DIR, 'sp500_adj_close.csv'), index_col=0, parse_dates=True)
    volumes = pd.read_csv(os.path.join(DATA_DIR, 'sp500_volume.csv'), index_col=0, parse_dates=True)
    market_caps = pd.read_csv(os.path.join(DATA_DIR, 'market_caps.csv'), index_col=0)

    windows = generate_windows(prices)
    selected_pairs_summary = []
    
    for w in tqdm(windows, desc="Windows"):
        fs, fe = w['formation_start'], w['formation_end']
        ts, te = w['trading_start'], w['trading_end']
        wid = w['window_id']
        
        try:
            universe = filter_universe(prices, volumes, market_caps, fs, fe)
            if len(universe) < 2:
                continue
                
            norm_prices = compute_normalized_prices(prices, universe, fs, fe)
            ssd_ranked = compute_ssd(norm_prices)
            ssd_map = {pair: val for pair, val in ssd_ranked}
            
            top_pairs = select_top_pairs(ssd_ranked, num_pairs=NUM_PAIRS)
            
            # Log Distance pairs
            for t1, t2 in top_pairs:
                selected_pairs_summary.append({
                    'window_id': wid, 'formation_start': fs.date(), 'formation_end': fe.date(),
                    'strategy': 'Distance', 'ticker_A': t1, 'ticker_B': t2, 
                    'ssd_value': ssd_map.get((t1, t2), 0.0)
                })
            
            # Log Cointegration pairs
            coint_pairs = find_cointegrated_pairs(prices, ssd_ranked, fs, fe, num_pairs=NUM_PAIRS, pvalue=COINT_PVALUE)
            for cp in coint_pairs:
                t1, t2 = cp['pair']
                selected_pairs_summary.append({
                    'window_id': wid, 'formation_start': fs.date(), 'formation_end': fe.date(),
                    'strategy': 'Cointegration', 'ticker_A': t1, 'ticker_B': t2,
                    'ssd_value': ssd_map.get((t1, t2), 0.0),
                    'coint_pvalue': cp['coint_pvalue'], 'beta': cp['beta']
                })
            
            # Log Copula pairs (uses same top_pairs as DM)
            for t1, t2 in top_pairs:
                selected_pairs_summary.append({
                    'window_id': wid, 'formation_start': fs.date(), 'formation_end': fe.date(),
                    'strategy': 'Copula', 'ticker_A': t1, 'ticker_B': t2,
                    'ssd_value': ssd_map.get((t1, t2), 0.0),
                    'copula_type': 't'
                })
                
        except Exception as e:
            continue

    df = pd.DataFrame(selected_pairs_summary)
    df.to_csv('outputs/tables/selected_pairs.csv', index=False)
    print(f"Done. Saved {len(df)} rows to outputs/tables/selected_pairs.csv")

if __name__ == "__main__":
    fix_selected_pairs()
