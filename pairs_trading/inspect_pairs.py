import pandas as pd
import os

path = 'outputs/tables/selected_pairs.csv'
if not os.path.exists(path):
    print(f"Error: {path} not found.")
else:
    sp = pd.read_csv(path)
    print('Total rows:', len(sp))
    print('Columns:', list(sp.columns))
    print()

    # Show pairs from first window for each strategy
    for strat in sp['strategy'].unique():
        subset = sp[(sp['window_id']==0) & (sp['strategy']==strat)]
        print(f'{strat} — Window 0 ({len(subset)} pairs):')
        for _, row in subset.iterrows():
            print(f'  {row["ticker_A"]:>5} - {row["ticker_B"]:<5}  SSD={row.get("ssd_value","N/A")}')
        print()
