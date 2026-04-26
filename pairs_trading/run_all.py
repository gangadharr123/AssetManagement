import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

from config import DATA_START, DATA_END, NUM_PAIRS, DM_THRESHOLD, COINT_THRESHOLD, COPULA_THRESHOLD, COINT_PVALUE, DATA_DIR, OUTPUT_DIR
from data.fetch_data import fetch_sp500_tickers, download_price_data, download_market_cap, download_risk_free_rate
from data.universe import filter_universe
from utils.windows import generate_windows
from utils.transaction_costs import apply_costs
from utils.portfolio_returns import compute_monthly_returns, compute_excess_returns
from strategies.pair_selection import compute_normalized_prices, compute_ssd, select_top_pairs
from strategies.distance_method import run_distance_method
from strategies.cointegration_method import find_cointegrated_pairs, run_cointegration_method
from strategies.copula_method import run_copula_method
from analysis.performance_metrics import compute_trade_statistics
from analysis.factor_regression import download_fama_french_factors

def main():
    print("========================================")
    print("Pairs Trading Research Pipeline")
    print("Survivor Bias: Using current S&P 500 components.")
    print("========================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
    
    try:
        if not os.path.exists(os.path.join(DATA_DIR, 'sp500_adj_close.csv')):
            tickers, _ = fetch_sp500_tickers()
            prices, volumes = download_price_data(tickers, DATA_START, DATA_END)
            market_caps = download_market_cap(tickers)
            rf_rates = download_risk_free_rate(DATA_START, DATA_END)
        else:
            print("Data already downloaded. Loading from disk...")
            prices = pd.read_csv(os.path.join(DATA_DIR, 'sp500_adj_close.csv'), index_col=0, parse_dates=True)
            volumes = pd.read_csv(os.path.join(DATA_DIR, 'sp500_volume.csv'), index_col=0, parse_dates=True)
            market_caps = pd.read_csv(os.path.join(DATA_DIR, 'market_caps.csv'), index_col=0)
            rf_rates = pd.read_csv(os.path.join(DATA_DIR, 'risk_free_daily.csv'), index_col=0, parse_dates=True)['Daily_RF']
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    windows = generate_windows(prices)
    
    all_trades = {'DM': [], 'Cointegration': [], 'Copula': []}
    universe_summary = []
    selected_pairs_summary = []
    
    print("\nRunning strategies across rolling windows...")
    for w in tqdm(windows, desc="Windows"):
        fs, fe = w['formation_start'], w['formation_end']
        ts, te = w['trading_start'], w['trading_end']
        wid = w['window_id']
        
        try:
            universe = filter_universe(prices, volumes, market_caps, fs, fe)
            
            universe_summary.append({
                'window_id': wid,
                'formation_start': fs.date(),
                'stocks_before_filter': len(prices.columns),
                'stocks_after_filter': len(universe)
            })
            
            if len(universe) < 2:
                continue
                
            norm_prices = compute_normalized_prices(prices, universe, fs, fe)
            ssd_ranked = compute_ssd(norm_prices)
            
            top_pairs = select_top_pairs(ssd_ranked, num_pairs=NUM_PAIRS)
            
            # Helper to find SSD for a pair
            ssd_map = {pair: val for pair, val in ssd_ranked}
            
            # Log Distance pairs
            for t1, t2 in top_pairs:
                selected_pairs_summary.append({
                    'window_id': wid, 'formation_start': fs.date(), 'formation_end': fe.date(),
                    'strategy': 'Distance', 'ticker_A': t1, 'ticker_B': t2, 
                    'ssd_value': ssd_map.get((t1, t2), 0.0)
                })
            
            dm_trades = run_distance_method(prices, top_pairs, fs, fe, ts, te, threshold=DM_THRESHOLD, window_id=wid)
            for t in dm_trades:
                t.update({'formation_start': fs.date(), 'trading_start': ts.date()})
            all_trades['DM'].extend(dm_trades)
            
            coint_pairs = find_cointegrated_pairs(prices, ssd_ranked, fs, fe, num_pairs=NUM_PAIRS, pvalue=COINT_PVALUE)
            
            # Log Cointegration pairs
            for cp in coint_pairs:
                t1, t2 = cp['pair']
                selected_pairs_summary.append({
                    'window_id': wid, 'formation_start': fs.date(), 'formation_end': fe.date(),
                    'strategy': 'Cointegration', 'ticker_A': t1, 'ticker_B': t2,
                    'ssd_value': ssd_map.get((t1, t2), 0.0),
                    'coint_pvalue': cp['coint_pvalue'], 'beta': cp['beta']
                })

            coint_trades = run_cointegration_method(prices, coint_pairs, ts, te, threshold=COINT_THRESHOLD, window_id=wid)
            for t in coint_trades:
                t.update({'formation_start': fs.date(), 'trading_start': ts.date()})
            all_trades['Cointegration'].extend(coint_trades)
            
            # For Copula, we use top_pairs. We need to fit them to get the copula type for logging.
            # In current implementation, it's always 't'. 
            for t1, t2 in top_pairs:
                selected_pairs_summary.append({
                    'window_id': wid, 'formation_start': fs.date(), 'formation_end': fe.date(),
                    'strategy': 'Copula', 'ticker_A': t1, 'ticker_B': t2,
                    'ssd_value': ssd_map.get((t1, t2), 0.0),
                    'copula_type': 't' # Current implementation default
                })

            copula_trades = run_copula_method(prices, top_pairs, fs, fe, ts, te, threshold=COPULA_THRESHOLD, window_id=wid)
            for t in copula_trades:
                t.update({'formation_start': fs.date(), 'trading_start': ts.date()})
            all_trades['Copula'].extend(copula_trades)
            
        except Exception as e:
            print(f"Error in window {wid}: {e}")
            continue

    print("\nApplying transaction costs...")
    for strat in all_trades:
        all_trades[strat] = apply_costs(all_trades[strat])
        
        # Save exact formatting for trades
        if all_trades[strat]:
            df_trades = pd.DataFrame(all_trades[strat])
            # format pair tuple to columns
            df_trades['ticker_A'] = df_trades['pair'].apply(lambda x: x[0])
            df_trades['ticker_B'] = df_trades['pair'].apply(lambda x: x[1])
            cols = ['ticker_A', 'ticker_B', 'entry_date', 'exit_date', 'direction',
                    'return_before_cost', 'return_after_cost', 'transaction_cost',
                    'converged', 'days_open', 'window_id', 'formation_start', 'trading_start']
            if strat == 'Cointegration':
                cols.append('beta')
            if strat == 'Copula':
                cols.append('copula_type')
                
            cols_avail = [c for c in cols if c in df_trades.columns]
            df_trades = df_trades[cols_avail]
            file_name = f'outputs/tables/{strat.lower()}_trades.csv'
            df_trades.to_csv(file_name, index=False)

    print("Computing portfolio returns...")
    monthly_rets = {}
    combined_monthly = None
    for strat, trades in all_trades.items():
        m_ret = compute_monthly_returns(trades, windows)
        exc_ret = compute_excess_returns(m_ret, rf_rates)
        monthly_rets[strat] = exc_ret
        
        if not exc_ret.empty:
            temp = exc_ret[['month', 'return_employed_before', 'return_employed_after', 'rf_rate']].copy()
            temp = temp.rename(columns={
                'return_employed_before': f'{strat.lower()}_ret_before',
                'return_employed_after': f'{strat.lower()}_ret_after'
            })
            if combined_monthly is None:
                combined_monthly = temp
            else:
                combined_monthly = combined_monthly.merge(temp.drop(columns=['rf_rate']), on='month', how='outer')

    if combined_monthly is not None:
        combined_monthly.to_csv('outputs/tables/monthly_returns.csv', index=False)

    pd.DataFrame(selected_pairs_summary).to_csv('outputs/tables/selected_pairs.csv', index=False)
    pd.DataFrame(universe_summary).to_csv('outputs/tables/universe_summary.csv', index=False)
    
    print("Downloading Fama French...")
    ff_factors = download_fama_french_factors()
    if not ff_factors.empty:
        ff_factors.to_csv('outputs/tables/fama_french_monthly.csv')
        
    print("Computing Summary Stats...")
    summary_records = []
    for strat, trades in all_trades.items():
        stats = compute_trade_statistics(trades)
        if not stats:
            continue
            
        c = stats.get('converged', {})
        u = stats.get('unconverged', {})
        
        exc_df = monthly_rets.get(strat, pd.DataFrame())
        mb = exc_df['return_employed_before'].mean() if not exc_df.empty else 0
        ma = exc_df['return_employed_after'].mean() if not exc_df.empty else 0
        s_std = exc_df['return_employed_after'].std() if not exc_df.empty else 0
        sh_b = (mb / exc_df['return_employed_before'].std() * np.sqrt(12)) if not exc_df.empty and exc_df['return_employed_before'].std() > 0 else 0
        sh_a = (ma / s_std * np.sqrt(12)) if s_std > 0 else 0
        
        summary_records.append({
            'strategy': strat,
            'total_trades': len(trades),
            'converged_pct': c.get('percentage', 0),
            'unconverged_pct': u.get('percentage', 0),
            'mean_monthly_ret_before': mb,
            'mean_monthly_ret_after': ma,
            'std_monthly': s_std,
            'sharpe_before': sh_b,
            'sharpe_after': sh_a,
            'avg_converged_return': c.get('mean_ret', 0),
            'avg_unconverged_return': u.get('mean_ret', 0),
            'avg_days_open': c.get('mean_days', 0)
        })
    pd.DataFrame(summary_records).to_csv('outputs/tables/summary_stats.csv', index=False)

    print("\n================ SUMMARY ================")
    for strat, exc_ret in monthly_rets.items():
        if exc_ret.empty:
            print(f"{strat}: No returns calculated.")
            continue
        mean_ret = exc_ret['return_employed_after'].mean() * 100
        std_ret = exc_ret['return_employed_after'].std()
        sharpe = (exc_ret['return_employed_after'].mean() / std_ret * np.sqrt(12)) if std_ret > 0 else 0
        print(f"{strat:15}: Mean Monthly Return = {mean_ret:.2f}%, Sharpe Ratio = {sharpe:.2f}")
    print("========================================")

if __name__ == "__main__":
    main()