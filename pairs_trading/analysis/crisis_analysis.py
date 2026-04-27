"""
CRISIS REGIME ANALYSIS MODULE
-----------------------------
This module identifies market regimes (Crisis vs. Normal) based on the annual returns
of the S&P 500 index. It computes performance metrics separately for each regime 
to test strategy robustness during market stress.
"""

import pandas as pd
import yfinance as yf
import sys
import os
import numpy as np

# Ensure config and performance metrics can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from analysis.performance_metrics import compute_all_metrics

def identify_crisis_periods(start, end):
    """
    Identifies 'Crisis' years within the sample period using S&P 500 annual returns.
    Following the paper's logic, crisis years are the bottom quintile (worst 20%) 
    of the annual returns in the study period.
    """
    print("Identifying crisis years using S&P 500 (^GSPC)...")
    data = yf.download("^GSPC", start=start, end=end, progress=False)
    
    # Handle both Series and DataFrame outputs from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']['^GSPC']
    else:
        prices = data['Close']
        
    # Re-sample to annual frequency and compute returns
    annual_prices = prices.resample('YE').last()
    annual_returns = annual_prices.pct_change().dropna()
    
    # Threshold for bottom 20%
    threshold = annual_returns.quantile(0.2)
    
    crisis_years = annual_returns[annual_returns <= threshold].index.year.tolist()
    normal_years = annual_returns[annual_returns > threshold].index.year.tolist()
    
    print(f"  Crisis years (worst 20%): {crisis_years}")
    return crisis_years, normal_years

def compare_crisis_normal(monthly_returns, crisis_years, normal_years):
    """
    Splits monthly strategy returns into Crisis and Normal subsamples and 
    compares their performance metrics.
    
    Returns:
        DataFrame: Comparison table of metrics across both regimes.
    """
    if monthly_returns.empty:
        return pd.DataFrame()
        
    df = monthly_returns.copy()
    df['year'] = df['month'].dt.year
    
    # Filter based on identified crisis years
    crisis_df = df[df['year'].isin(crisis_years)]
    normal_df = df[df['year'].isin(normal_years)]
    
    # Compute metrics for each group
    metrics_crisis = compute_all_metrics(crisis_df['excess_return_after'])
    metrics_normal = compute_all_metrics(normal_df['excess_return_after'])
    
    # Define primary metrics for comparison
    keys = ['sharpe', 'sortino', 'mean', 'cvar_95', 'max_drawdown']
    
    comparison_data = []
    for k in keys:
        comparison_data.append({
            'Metric': k,
            'Crisis Regime': metrics_crisis.get(k, np.nan),
            'Normal Regime': metrics_normal.get(k, np.nan)
        })
        
    return pd.DataFrame(comparison_data)
