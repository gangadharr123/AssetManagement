import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(prices, ssd_ranked_pairs, formation_start,
                            formation_end, num_pairs=20, pvalue=0.05):
    coint_pairs = []
    form_prices = prices.loc[formation_start:formation_end]
    
    for item in ssd_ranked_pairs:
        if len(item) == 2 and isinstance(item[0], tuple):
            t1, t2 = item[0]
        else:
            t1, t2 = item
            
        p1 = form_prices[t1]
        p2 = form_prices[t2]
        
        try:
            _, pval, _ = coint(p1, p2)
        except Exception:
            continue
            
        if pval < pvalue:
            X = sm.add_constant(p1)
            model = sm.OLS(p2, X).fit()
            beta = model.params[t1]
            
            spread = p2 - beta * p1
            
            coint_pairs.append({
                'pair': (t1, t2),
                'beta': beta,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'coint_pvalue': pval
            })
            
            if len(coint_pairs) >= num_pairs:
                break
                
    return coint_pairs

def run_cointegration_method(prices, coint_pairs, trading_start, trading_end,
                             threshold=2.0, window_id=0):
    trades = []
    trade_prices = prices.loc[trading_start:trading_end]
    if trade_prices.empty:
        return trades
        
    for cp in coint_pairs:
        t1, t2 = cp['pair']
        beta = cp['beta']
        mean = cp['spread_mean']
        std = cp['spread_std']
        
        p1 = trade_prices[t1]
        p2 = trade_prices[t2]
        
        spread = p2 - beta * p1
        norm_spread = (spread - mean) / std
        
        position = 0
        entry_date = None
        entry_p1 = 0
        entry_p2 = 0
        
        for date, ns in norm_spread.items():
            if position == 0:
                if ns > threshold:
                    position = -1 # Sell B, Buy A
                    entry_date = date
                    entry_p1 = p1.loc[date]
                    entry_p2 = p2.loc[date]
                elif ns < -threshold:
                    position = 1 # Buy B, Sell A
                    entry_date = date
                    entry_p1 = p1.loc[date]
                    entry_p2 = p2.loc[date]
            else:
                converged = False
                if position == -1 and ns <= 0:
                    converged = True
                elif position == 1 and ns >= 0:
                    converged = True
                    
                force_close = (date == trade_prices.index[-1])
                
                if converged or force_close:
                    exit_p1 = p1.loc[date]
                    exit_p2 = p2.loc[date]
                    
                    ret1 = exit_p1 / entry_p1 - 1
                    ret2 = exit_p2 / entry_p2 - 1
                    
                    if position == 1:
                        # Buy $1 of B, short $beta of A
                        profit = 1.0 * ret2 - beta * ret1
                        direction = 'long_B_short_A'
                    else:
                        # Buy $1 of A, short $1/beta of B
                        profit = 1.0 * ret1 - (1.0 / beta) * ret2
                        direction = 'short_B_long_A'
                        
                    ret = profit / 1.0 
                    
                    trades.append({
                        'pair': (t1, t2),
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': direction,
                        'return_before_cost': ret,
                        'converged': converged,
                        'days_open': (date - entry_date).days,
                        'window_id': window_id,
                        'beta': beta
                    })
                    position = 0
                    
    return trades