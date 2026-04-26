import pandas as pd
import numpy as np

def run_distance_method(prices, pairs, formation_start, formation_end,
                        trading_start, trading_end, threshold=2.0, window_id=0):
    trades = []
    
    form_prices = prices.loc[formation_start:formation_end]
    norm_form_prices = form_prices / form_prices.iloc[0]
    
    spread_stds = {}
    for t1, t2 in pairs:
        spread = norm_form_prices[t1] - norm_form_prices[t2]
        spread_stds[(t1, t2)] = spread.std()
        
    trade_prices = prices.loc[trading_start:trading_end]
    if trade_prices.empty:
        return trades
        
    norm_trade_prices = trade_prices / trade_prices.iloc[0]
    
    for t1, t2 in pairs:
        spread = norm_trade_prices[t1] - norm_trade_prices[t2]
        std = spread_stds[(t1, t2)]
        
        position = 0 
        entry_date = None
        entry_price_1 = 0
        entry_price_2 = 0
        
        for date, sp in spread.items():
            if position == 0:
                if sp > threshold * std:
                    position = -1
                    entry_date = date
                    entry_price_1 = trade_prices.at[date, t1]
                    entry_price_2 = trade_prices.at[date, t2]
                elif sp < -threshold * std:
                    position = 1
                    entry_date = date
                    entry_price_1 = trade_prices.at[date, t1]
                    entry_price_2 = trade_prices.at[date, t2]
            else:
                converged = False
                if position == -1 and sp <= 0:
                    converged = True
                elif position == 1 and sp >= 0:
                    converged = True
                    
                force_close = (date == trade_prices.index[-1])
                
                if converged or force_close:
                    exit_price_1 = trade_prices.at[date, t1]
                    exit_price_2 = trade_prices.at[date, t2]
                    
                    ret_1 = exit_price_1 / entry_price_1 - 1
                    ret_2 = exit_price_2 / entry_price_2 - 1
                    
                    if position == 1:
                        ret = ret_1 - ret_2
                        direction = 'long_A_short_B'
                    else:
                        ret = ret_2 - ret_1
                        direction = 'short_A_long_B'
                        
                    trades.append({
                        'pair': (t1, t2),
                        'entry_date': entry_date,
                        'exit_date': date,
                        'return_before_cost': ret,
                        'direction': direction,
                        'converged': converged,
                        'days_open': (date - entry_date).days,
                        'window_id': window_id
                    })
                    position = 0
                    
    return trades