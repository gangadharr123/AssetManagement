import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_windows(prices):
    windows = []
    
    start_date = prices.index[0]
    end_date = prices.index[-1]
    
    current_form_start = start_date
    window_id = 0
    
    while True:
        # Formation period: 12 months
        form_end_target = current_form_start + MonthEnd(config.FORMATION_MONTHS)
        
        # Trading period: next 6 months
        trade_start_target = form_end_target + pd.Timedelta(days=1)
        trade_end_target = trade_start_target + MonthEnd(config.TRADING_MONTHS)
        
        if trade_end_target > end_date:
            break
            
        # Map to nearest actual trading days
        try:
            form_end = prices.index[prices.index <= form_end_target][-1]
            trade_start = prices.index[prices.index >= trade_start_target][0]
            trade_end = prices.index[prices.index <= trade_end_target][-1]
        except IndexError:
            break
        
        if form_end >= trade_start:
            # Prevent overlap
            trade_start_candidates = prices.index[prices.index > form_end]
            if len(trade_start_candidates) == 0:
                break
            trade_start = trade_start_candidates[0]
            
        windows.append({
            'formation_start': current_form_start,
            'formation_end': form_end,
            'trading_start': trade_start,
            'trading_end': trade_end,
            'window_id': window_id
        })
        
        # Roll forward by 1 calendar month
        current_form_start = current_form_start + MonthBegin(1)
        
        # Snap to actual trading day
        try:
            current_form_start = prices.index[prices.index >= current_form_start][0]
        except IndexError:
            break
            
        window_id += 1
        
    print(f"Generated {len(windows)} rolling windows from {windows[0]['formation_start'].date()} to {windows[-1]['trading_end'].date()}")
    return windows