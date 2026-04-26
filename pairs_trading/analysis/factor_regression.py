import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import requests
import zipfile
import io
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def download_fama_french_factors():
    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    def process_zip(url, cols):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            lines = f.read().decode('utf-8').splitlines()
            
        start = next(i for i, l in enumerate(lines) if l.strip() and l.strip()[0].isdigit())
        
        # Filter for valid monthly data rows (first element must be 6 digits YYYYMM)
        data = []
        for l in lines[start:]:
            s = l.strip()
            if s and s[0].isdigit() and len(s.split(',')[0].strip()) == 6:
                data.append(s)
                
        df = pd.read_csv(io.StringIO('\n'.join(data)), header=None, names=cols)
        df[cols[0]] = pd.to_datetime(df[cols[0]].astype(str).str.strip(), format='%Y%m')
        for c in cols[1:]:
            df[c] = df[c] / 100.0
        return df

    try:
        ff5 = process_zip(ff5_url, ['month','Mkt_RF','SMB','HML','RMW','CMA','RF'])
        mom = process_zip(mom_url, ['month','MOM'])
        
        merged = ff5.merge(mom, on='month')
        merged.to_csv(os.path.join(config.DATA_DIR, 'ff_factors_monthly.csv'), index=False)
        return merged
    except Exception as e:
        print(f"Error downloading factors: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

def run_factor_regressions(monthly_returns, factors):
    results = {}
    
    if factors.empty or monthly_returns.empty:
        return results
        
    df = monthly_returns.copy()
    df = df.set_index('month')
    
    df.index = df.index.to_period('M').to_timestamp()
    factors.index = factors.index.to_period('M').to_timestamp()
    
    merged = df.join(factors, how='inner').dropna()
    
    if merged.empty:
        return results
        
    y = merged['excess_return_after']
    
    mom_col = [c for c in merged.columns if 'Mom' in c][0]
    X1 = merged[['Mkt-RF', 'SMB', 'HML', mom_col]]
    X2 = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X3 = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', mom_col]]
    
    models = {'FF3+Mom': X1, 'FF5': X2, 'FF5+Mom': X3}
    
    for name, X in models.items():
        try:
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': config.NEWEY_WEST_LAGS})
            
            res = {
                'alpha': model.params['const'],
                'alpha_t': model.tvalues['const'],
                'R_squared': model.rsquared
            }
            
            for col in X.columns:
                res[f'{col}_beta'] = model.params[col]
                res[f'{col}_t'] = model.tvalues[col]
                
            results[name] = res
        except Exception as e:
            print(f"Regression {name} failed: {e}")
            
    return results