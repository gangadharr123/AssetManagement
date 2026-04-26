import pandas as pd
import numpy as np

def compute_normalized_prices(prices, tickers, start, end):
    subset = prices.loc[start:end, tickers]
    normalized = subset / subset.iloc[0]
    return normalized

def compute_ssd(normalized_prices):
    arr = normalized_prices.values
    tickers = normalized_prices.columns
    
    n = arr.shape[1]
    ssd_dict = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = arr[:, i] - arr[:, j]
            ssd_dict[(tickers[i], tickers[j])] = np.sum(diff ** 2)
            
    sorted_pairs = sorted(ssd_dict.items(), key=lambda item: item[1])
    return sorted_pairs

def select_top_pairs(ssd_rankings, num_pairs=20):
    return [pair[0] for pair in ssd_rankings[:num_pairs]]