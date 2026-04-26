import pandas as pd
import yfinance as yf
import os
import sys
import time
import random

# Ensure config can be imported when running as a module or script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    import requests
    import io
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"Fetched {len(tickers)} S&P 500 tickers")
    return tickers, df

def download_price_data(tickers, start, end):
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end} in batches...")
    
    all_prices = []
    all_volumes = []
    
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1}/{len(tickers)//batch_size + 1}: {batch[:3]}...")
        
        for attempt in range(3):
            try:
                # yfinance >= 0.2.30 handles user-agent automatically using curl_cffi
                data = yf.download(batch, start=start, end=end, group_by='ticker', threads=True, progress=False)
                
                if not data.empty:
                    batch_prices = pd.DataFrame()
                    batch_volumes = pd.DataFrame()
                    
                    if len(batch) == 1:
                        if 'Close' in data:
                            batch_prices[batch[0]] = data['Close']
                        if 'Volume' in data:
                            batch_volumes[batch[0]] = data['Volume']
                    else:
                        # yf.download returns a MultiIndex DataFrame with tickers on level 0 or 1
                        for ticker in batch:
                            if isinstance(data.columns, pd.MultiIndex):
                                if ticker in data.columns.levels[0]:
                                    if 'Close' in data[ticker]:
                                        batch_prices[ticker] = data[ticker]['Close']
                                    if 'Volume' in data[ticker]:
                                        batch_volumes[ticker] = data[ticker]['Volume']
                                elif ticker in data.columns.levels[1]:
                                    # newer yfinance versions put price fields on level 0, tickers on level 1
                                    if 'Close' in data.columns.levels[0] and ticker in data['Close']:
                                        batch_prices[ticker] = data['Close'][ticker]
                                    if 'Volume' in data.columns.levels[0] and ticker in data['Volume']:
                                        batch_volumes[ticker] = data['Volume'][ticker]
                                        
                    if not batch_prices.empty:
                        all_prices.append(batch_prices)
                    if not batch_volumes.empty:
                        all_volumes.append(batch_volumes)
                    break
            except Exception as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt+1}/3 after {wait:.1f}s: {e}")
                time.sleep(wait)
                
        time.sleep(random.uniform(1, 3))

    if not all_prices:
        print("Failed to download any data.")
        return pd.DataFrame(), pd.DataFrame()
        
    prices = pd.concat(all_prices, axis=1)
    volumes = pd.concat(all_volumes, axis=1)

    # Clean duplicate columns if any
    prices = prices.loc[:, ~prices.columns.duplicated()]
    volumes = volumes.loc[:, ~volumes.columns.duplicated()]

    valid_tickers = []
    for ticker in prices.columns:
        if prices[ticker].isna().mean() <= 0.20:
            valid_tickers.append(ticker)
    
    prices = prices[valid_tickers].ffill(limit=5)
    volumes = volumes[valid_tickers].ffill(limit=5)
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    prices.to_csv(os.path.join(config.DATA_DIR, 'sp500_adj_close.csv'))
    volumes.to_csv(os.path.join(config.DATA_DIR, 'sp500_volume.csv'))
    
    print(f"Downloaded data for {len(valid_tickers)} tickers, {len(prices)} trading days")
    return prices, volumes

def download_market_cap(tickers):
    market_caps = {}
    print(f"Retrieving market caps for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            if 'marketCap' in info and info['marketCap'] is not None:
                market_caps[ticker] = info['marketCap']
            time.sleep(random.uniform(0.1, 0.5)) # Be gentle
        except Exception:
            time.sleep(random.uniform(1, 2))
            continue
        if (i + 1) % 50 == 0:
            print(f"Retrieved market cap for {len(market_caps)}/{i + 1} tickers")
            
    df = pd.Series(market_caps, name='MarketCap').to_frame()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.DATA_DIR, 'market_caps.csv'))
    print(f"Total retrieved market caps: {len(market_caps)}")
    return df

def download_risk_free_rate(start, end):
    data = yf.download(config.RISK_FREE_TICKER, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.levels[0]:
            yield_pct = data['Close'][config.RISK_FREE_TICKER]
        else:
            yield_pct = data.iloc[:, 0]
    elif 'Close' in data.columns:
        yield_pct = data['Close']
    else:
        yield_pct = data.iloc[:, 0]
        
    daily_rf = (1 + yield_pct / 100) ** (1 / 252) - 1
    daily_rf.name = 'Daily_RF'
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    daily_rf.to_csv(os.path.join(config.DATA_DIR, 'risk_free_daily.csv'))
    return daily_rf
