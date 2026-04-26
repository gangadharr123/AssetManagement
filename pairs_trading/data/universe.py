import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def filter_universe(prices, volume, market_caps, formation_start, formation_end):
    form_prices = prices.loc[formation_start:formation_end]
    form_volume = volume.loc[formation_start:formation_end]
    
    n_start = len(form_prices.columns)
    
    # 1. No NaN in formation period
    valid_cols = form_prices.dropna(axis=1).columns
    form_prices = form_prices[valid_cols]
    form_volume = form_volume[valid_cols]
    
    # 2. No days with volume = 0
    # Avoid ValueError by ensuring columns match before comparison
    common_cols = form_prices.columns.intersection(form_volume.columns)
    form_prices = form_prices[common_cols]
    form_volume = form_volume[common_cols]
    
    valid_cols = form_volume.columns[(form_volume > 0).all()]
    form_prices = form_prices[valid_cols]
    
    # 3. Average price >= MIN_PRICE
    valid_cols = form_prices.columns[form_prices.mean() >= config.MIN_PRICE]
    form_prices = form_prices[valid_cols]
    
    # 4. Remove bottom MKTCAP_BOTTOM_DECILE
    available_mcaps = market_caps.reindex(valid_cols).dropna()['MarketCap']
    if len(available_mcaps) > 0:
        threshold = available_mcaps.quantile(config.MKTCAP_BOTTOM_DECILE)
        valid_cols = available_mcaps[available_mcaps >= threshold].index.tolist()
    else:
        valid_cols = form_prices.columns.tolist()
        
    n_end = len(valid_cols)
    print(f"Universe filtered: {n_start} -> {n_end} stocks for {formation_start.date()} to {formation_end.date()}")
    
    return valid_cols