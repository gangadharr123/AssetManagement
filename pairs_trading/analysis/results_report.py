import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from analysis.performance_metrics import compute_trade_statistics, compute_all_metrics

def ensure_dirs():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.TABLES_DIR, exist_ok=True)

def generate_monthly_excess_returns_table(rets_dict):
    ensure_dirs()
    records = []
    
    for name, df in rets_dict.items():
        if df.empty:
            continue
            
        mb = compute_all_metrics(df['excess_return_before'])
        ma = compute_all_metrics(df['excess_return_after'])
        
        records.append({
            'Strategy': name,
            'Cost': 'Before',
            'Mean': mb.get('mean'),
            't-stat': mb.get('t_stat'),
            'Std': mb.get('std'),
            'Sharpe': mb.get('sharpe'),
            'Skew': mb.get('skewness'),
            'Kurt': mb.get('kurtosis'),
            'VaR(5%)': mb.get('var_95'),
            'CVaR(5%)': mb.get('cvar_95')
        })
        records.append({
            'Strategy': name,
            'Cost': 'After',
            'Mean': ma.get('mean'),
            't-stat': ma.get('t_stat'),
            'Std': ma.get('std'),
            'Sharpe': ma.get('sharpe'),
            'Skew': ma.get('skewness'),
            'Kurt': ma.get('kurtosis'),
            'VaR(5%)': ma.get('var_95'),
            'CVaR(5%)': ma.get('cvar_95')
        })
        
    res_df = pd.DataFrame(records)
    res_df.to_csv(os.path.join(config.TABLES_DIR, 'monthly_excess_returns.csv'), index=False)
    return res_df

def generate_risk_adjusted_performance_table(rets_dict):
    ensure_dirs()
    records = []
    for name, df in rets_dict.items():
        if df.empty:
            continue
        m = compute_all_metrics(df['excess_return_after'])
        records.append({
            'Strategy': name,
            'Omega': m.get('omega'),
            'Sortino': m.get('sortino'),
            'Kappa3': np.nan,
            'MaxDD': m.get('max_drawdown'),
            'Calmar': m.get('calmar'),
            'Sterling': np.nan,
            'Burke': np.nan
        })
    res_df = pd.DataFrame(records)
    res_df.to_csv(os.path.join(config.TABLES_DIR, 'risk_adjusted_performance.csv'), index=False)
    
def generate_trade_properties_table(trades_dict):
    ensure_dirs()
    records = []
    for name, trades in trades_dict.items():
        stats = compute_trade_statistics(trades)
        if not stats:
            continue
            
        c = stats.get('converged', {})
        u = stats.get('unconverged', {})
        
        records.append({
            'Strategy': name,
            'Conv_%': c.get('percentage'),
            'Conv_Mean_Ret': c.get('mean_ret'),
            'Conv_Days': c.get('mean_days'),
            'Conv_Pct_Pos': c.get('pct_positive'),
            'Unconv_%': u.get('percentage'),
            'Unconv_Mean_Ret': u.get('mean_ret')
        })
        
    res_df = pd.DataFrame(records)
    res_df.to_csv(os.path.join(config.TABLES_DIR, 'trade_properties.csv'), index=False)
    
def generate_factor_regression_table(reg_results_dict):
    ensure_dirs()
    records = []
    for strategy, models in reg_results_dict.items():
        for model_name, res in models.items():
            row = {'Strategy': strategy, 'Model': model_name}
            row.update(res)
            records.append(row)
            
    res_df = pd.DataFrame(records)
    res_df.to_csv(os.path.join(config.TABLES_DIR, 'factor_regression_results.csv'), index=False)

def plot_cumulative_returns(rets_dict):
    ensure_dirs()
    plt.figure(figsize=(10, 6))
    
    colors = {'DM': 'blue', 'Cointegration': 'orange', 'Copula': 'green'}
    styles = {'DM': '-', 'Cointegration': '-', 'Copula': '--'}
    
    for name, df in rets_dict.items():
        if df.empty:
            continue
            
        df = df.set_index('month')
        cum_ret = (1 + df['excess_return_after']).cumprod()
        plt.plot(cum_ret.index, cum_ret.values, label=name, 
                 color=colors.get(name, 'black'), linestyle=styles.get(name, '-'))
                 
    plt.yscale('log')
    plt.title('Cumulative Excess Returns ($1 Invested)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'cumulative_excess_return.png'), dpi=150)
    plt.close()

def plot_rolling_sharpe(rets_dict):
    ensure_dirs()
    plt.figure(figsize=(10, 6))
    window = 60
    
    for name, df in rets_dict.items():
        if df.empty or len(df) < window:
            continue
            
        df = df.set_index('month')
        rolling_mean = df['excess_return_after'].rolling(window).mean()
        rolling_std = df['excess_return_after'].rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
        
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=name)
        
    plt.title(f'{window}-Month Rolling Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'rolling_sharpe_5yr.png'), dpi=150)
    plt.close()

def plot_trade_distributions(trades_dict):
    ensure_dirs()
    names = {'DM': 'dm', 'Cointegration': 'coint', 'Copula': 'copula'}
    
    for name, trades in trades_dict.items():
        if not trades:
            continue
            
        rets = [t['return_before_cost'] for t in trades]
        plt.figure(figsize=(8, 5))
        plt.hist(rets, bins=50, alpha=0.7, color='blue')
        plt.title(f'Trade Return Distribution - {name}')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fname = names.get(name, name.lower())
        plt.savefig(os.path.join(config.FIGURES_DIR, f'trade_distribution_{fname}.png'), dpi=150)
        plt.close()

def generate_all_reports(rets_dict, trades_dict, reg_results_dict, sensitivity_dict, crisis_df):
    ensure_dirs()
    
    generate_monthly_excess_returns_table(rets_dict)
    generate_risk_adjusted_performance_table(rets_dict)
    generate_trade_properties_table(trades_dict)
    generate_factor_regression_table(reg_results_dict)
    
    if 'Copula' in sensitivity_dict:
        sensitivity_dict['Copula'].to_csv(os.path.join(config.TABLES_DIR, 'sensitivity_copula.csv'))
        
    if not crisis_df.empty:
        crisis_df.to_csv(os.path.join(config.TABLES_DIR, 'crisis_vs_normal.csv'), index=False)
        
    plot_cumulative_returns(rets_dict)
    plot_rolling_sharpe(rets_dict)
    plot_trade_distributions(trades_dict)
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'Positive/Negative Trades (Placeholder)', ha='center')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'positive_negative_trades.png'), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'Threshold Sensitivity (Placeholder)', ha='center')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'threshold_sensitivity.png'), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'Copula Cumulative vs Thresholds (Placeholder)', ha='center')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'copula_cumulative_thresholds.png'), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'Crisis vs Normal (Placeholder)', ha='center')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'crisis_performance.png'), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'Subperiod Performance (Placeholder)', ha='center')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'subperiod_performance.png'), dpi=150)
    plt.close()