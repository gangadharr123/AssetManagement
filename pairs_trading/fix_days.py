import pandas as pd
import numpy as np

for fname in ['dm_trades.csv', 'cointegration_trades.csv', 'copula_trades.csv']:
    path = f'outputs/tables/{fname}'
    df = pd.read_csv(path)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['days_open'] = df.apply(
        lambda r: np.busday_count(r['entry_date'].date(), r['exit_date'].date()), axis=1)
    df.to_csv(path, index=False)
    conv = df[df['converged']==True]
    print(f"{fname}: Avg trading days = {conv['days_open'].mean():.1f}, Converged = {len(conv)}/{len(df)}")