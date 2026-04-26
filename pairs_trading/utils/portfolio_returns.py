import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def compute_monthly_returns(all_trades, windows):
    if not all_trades:
        return pd.DataFrame()
        
    trades_df = pd.DataFrame(all_trades)
    
    min_date = trades_df['entry_date'].min()
    max_date = trades_df['exit_date'].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame()
        
    months = pd.date_range(start=min_date.replace(day=1), end=max_date, freq='MS')
    
    results = []
    
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(1)
        
        # Active portfolios (windows) this month
        active_windows = [w['window_id'] for w in windows if w['trading_start'] <= month_end and w['trading_end'] >= month_start]
        
        if not active_windows:
            continue
            
        # Find trades open during this month
        open_trades = trades_df[
            (trades_df['entry_date'] <= month_end) & 
            (trades_df['exit_date'] >= month_start)
        ].copy()
        
        if open_trades.empty:
            results.append({
                'month': month_start,
                'return_employed_before': 0.0,
                'return_employed_after': 0.0,
                'return_committed_before': 0.0,
                'return_committed_after': 0.0,
                'num_active_portfolios': len(active_windows),
                'num_trades': 0
            })
            continue
            
        open_trades['days_in_month'] = open_trades.apply(
            lambda row: max(0, (min(row['exit_date'], month_end) - max(row['entry_date'], month_start)).days + 1),
            axis=1
        )
        open_trades['total_days'] = open_trades.apply(
            lambda row: max(1, (row['exit_date'] - row['entry_date']).days + 1),
            axis=1
        )
        open_trades['mtm_ratio'] = open_trades['days_in_month'] / open_trades['total_days']
        
        open_trades['ret_before_mtm'] = open_trades['return_before_cost'] * open_trades['mtm_ratio']
        open_trades['ret_after_mtm'] = open_trades['return_after_cost'] * open_trades['mtm_ratio']
        
        monthly_port_rets_before = []
        monthly_port_rets_after = []
        
        for w_id in active_windows:
            w_trades = open_trades[open_trades['window_id'] == w_id]
            
            if w_trades.empty:
                monthly_port_rets_before.append(0.0)
                monthly_port_rets_after.append(0.0)
            else:
                total_ret_before = w_trades['ret_before_mtm'].sum()
                total_ret_after = w_trades['ret_after_mtm'].sum()
                
                monthly_port_rets_before.append(total_ret_before / config.NUM_PAIRS)
                monthly_port_rets_after.append(total_ret_after / config.NUM_PAIRS)
                
        results.append({
            'month': month_start,
            'return_employed_before': np.mean(monthly_port_rets_before) if monthly_port_rets_before else 0.0,
            'return_employed_after': np.mean(monthly_port_rets_after) if monthly_port_rets_after else 0.0,
            'return_committed_before': np.mean(monthly_port_rets_before),
            'return_committed_after': np.mean(monthly_port_rets_after),
            'num_active_portfolios': len(active_windows),
            'num_trades': len(open_trades)
        })
        
    return pd.DataFrame(results)

def compute_excess_returns(monthly_returns, risk_free_rates):
    if monthly_returns.empty:
        return monthly_returns
        
    rf_monthly = risk_free_rates.resample('MS').mean() * 21 # Approx 21 trading days in month
    
    monthly_returns = monthly_returns.set_index('month')
    monthly_returns['rf_rate'] = rf_monthly.reindex(monthly_returns.index).fillna(0)
    
    monthly_returns['excess_return_before'] = monthly_returns['return_committed_before'] - monthly_returns['rf_rate']
    monthly_returns['excess_return_after'] = monthly_returns['return_committed_after'] - monthly_returns['rf_rate']
    
    return monthly_returns.reset_index()