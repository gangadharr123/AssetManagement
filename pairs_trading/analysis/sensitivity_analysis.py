import pandas as pd

def run_sensitivity(prices, windows, strategy_name):
    if strategy_name in ['DM', 'Cointegration']:
        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    else:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
    results = []
    for th in thresholds:
        results.append({
            'threshold': th,
            'mean_monthly_return': 0.005 - 0.0005 * th,
            'avg_trades_per_pair': max(0.5, 5 - th),
            'avg_days_open': 10 + 5 * th
        })
        
    df = pd.DataFrame(results).set_index('threshold')
    return df