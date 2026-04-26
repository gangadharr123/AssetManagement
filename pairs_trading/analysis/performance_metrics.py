import numpy as np
import pandas as pd
import scipy.stats as stats

def compute_all_metrics(monthly_excess_returns_series):
    if len(monthly_excess_returns_series) == 0:
        return {}
        
    ret = monthly_excess_returns_series
    
    mean = ret.mean()
    std = ret.std()
    
    n = len(ret)
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
    
    sharpe = (mean / std) * np.sqrt(12) if std > 0 else 0
    
    downside = ret[ret < 0]
    sortino = (mean / downside.std()) * np.sqrt(12) if len(downside) > 0 and downside.std() > 0 else 0
    
    skewness = stats.skew(ret)
    kurtosis = stats.kurtosis(ret)
    
    var_95 = np.percentile(ret, 5)
    cvar_95 = ret[ret <= var_95].mean()
    
    cum_ret = (1 + ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown = drawdown.min()
    
    calmar = (mean * 12) / abs(max_drawdown) if max_drawdown < 0 else np.nan
    
    gains = ret[ret > 0].sum()
    losses = abs(ret[ret < 0].sum())
    omega = gains / losses if losses > 0 else np.nan
    
    return {
        'mean': mean,
        'std': std,
        't_stat': t_stat,
        'sharpe': sharpe,
        'sortino': sortino,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'omega': omega
    }

def compute_trade_statistics(trades):
    if not trades:
        return {}
        
    df = pd.DataFrame(trades)
    
    def get_stats(subset):
        if subset.empty:
            return {'count': 0}
        
        rets = subset['return_before_cost']
        return {
            'count': len(subset),
            'percentage': len(subset) / len(df) * 100,
            'mean_ret': rets.mean(),
            'std_ret': rets.std(),
            'sharpe': (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0,
            'skewness': stats.skew(rets) if len(rets) > 1 else 0,
            'mean_days': subset['days_open'].mean(),
            'median_days': subset['days_open'].median(),
            'pct_positive': (rets > 0).mean() * 100,
            'distinct_pairs': subset['pair'].nunique()
        }
        
    converged = df[df['converged'] == True]
    unconverged = df[df['converged'] == False]
    
    return {
        'converged': get_stats(converged),
        'unconverged': get_stats(unconverged),
        'total_distinct_pairs': df['pair'].nunique()
    }