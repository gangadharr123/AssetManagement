import pandas as pd
import yfinance as yf
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from analysis.performance_metrics import compute_all_metrics

def identify_crisis_periods(start, end):
    print("Identifying crisis years using ^GSPC...")
    data = yf.download("^GSPC", start=start, end=end)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']['^GSPC']
    else:
        prices = data['Close']
        
    annual_prices = prices.resample('YE').last()
    annual_returns = annual_prices.pct_change().dropna()
    
    threshold = annual_returns.quantile(0.2)
    
    crisis_years = annual_returns[annual_returns <= threshold].index.year.tolist()
    normal_years = annual_returns[annual_returns > threshold].index.year.tolist()
    
    print(f"Crisis years identified: {crisis_years}")
    return crisis_years, normal_years

def compare_crisis_normal(monthly_returns, crisis_years, normal_years):
    if monthly_returns.empty:
        return pd.DataFrame()
        
    df = monthly_returns.copy()
    df['year'] = df['month'].dt.year
    
    crisis_df = df[df['year'].isin(crisis_years)]
    normal_df = df[df['year'].isin(normal_years)]
    
    metrics_crisis = compute_all_metrics(crisis_df['excess_return_after'])
    metrics_normal = compute_all_metrics(normal_df['excess_return_after'])
    
    keys = ['sharpe', 'sortino', 'mean', 'cvar_95', 'max_drawdown']
    
    comp = []
    for k in keys:
        comp.append({
            'Metric': k,
            'Crisis': metrics_crisis.get(k, np.nan),
            'Normal': metrics_normal.get(k, np.nan)
        })
        
    return pd.DataFrame(comp)