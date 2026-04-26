import pandas as pd

# Load DM trades
dm = pd.read_csv('outputs/tables/dm_trades.csv')
print("DM TRADES DIAGNOSIS")
print(f"Total trades: {len(dm)}")
print(f"Converged: {dm['converged'].sum()} ({dm['converged'].mean()*100:.1f}%)")
print(f"Unconverged: {(~dm['converged']).sum()} ({(~dm['converged']).mean()*100:.1f}%)")
print(f"\nConverged trades:")
conv = dm[dm['converged']==True]
print(f"  Mean return before cost: {conv['return_before_cost'].mean()*100:.2f}%")
print(f"  Mean return after cost:  {conv['return_after_cost'].mean()*100:.2f}%")
print(f"  Pct positive: {(conv['return_before_cost']>0).mean()*100:.1f}%")
print(f"  Avg days open: {conv['days_open'].mean():.1f}")
print(f"\nUnconverged trades:")
unconv = dm[dm['converged']==False]
print(f"  Mean return before cost: {unconv['return_before_cost'].mean()*100:.2f}%")
print(f"  Mean return after cost:  {unconv['return_after_cost'].mean()*100:.2f}%")

print(f"\nSample of 10 converged DM trades:")
print(conv[['ticker_A','ticker_B','entry_date','exit_date','return_before_cost','days_open']].head(10).to_string())

# Same for Cointegration
print("\n\nCOINTEGRATION TRADES DIAGNOSIS")
ci = pd.read_csv('outputs/tables/cointegration_trades.csv')
print(f"Total trades: {len(ci)}")
print(f"Converged: {ci['converged'].sum()} ({ci['converged'].mean()*100:.1f}%)")
conv_ci = ci[ci['converged']==True]
unconv_ci = ci[ci['converged']==False]
print(f"Converged mean return: {conv_ci['return_before_cost'].mean()*100:.2f}%")
print(f"Unconverged mean return: {unconv_ci['return_before_cost'].mean()*100:.2f}%")

# Same for Copula
print("\n\nCOPULA TRADES DIAGNOSIS")  
co = pd.read_csv('outputs/tables/copula_trades.csv')
print(f"Total trades: {len(co)}")
print(f"Converged: {co['converged'].sum()} ({co['converged'].mean()*100:.1f}%)")
conv_co = co[co['converged']==True]
unconv_co = co[co['converged']==False]
print(f"Converged mean return: {conv_co['return_before_cost'].mean()*100:.2f}%")
print(f"Unconverged mean return: {unconv_co['return_before_cost'].mean()*100:.2f}%")

# Check monthly returns
print("\n\nMONTHLY RETURNS")
mr = pd.read_csv('outputs/tables/monthly_returns.csv')
print(f"Months with data: {len(mr)}")
print(mr.describe().round(4).to_string())