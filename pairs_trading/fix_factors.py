import pandas as pd
import requests
import zipfile
import io

def download_ff(url, cols):
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

ff5 = download_ff(
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    ['month','Mkt_RF','SMB','HML','RMW','CMA','RF'])
    
mom = download_ff(
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
    ['month','MOM'])
    
factors = ff5.merge(mom, on='month')
factors.to_csv('outputs/tables/fama_french_monthly.csv', index=False)

print(f"Saved {len(factors)} months of factor data")
print(factors.tail())