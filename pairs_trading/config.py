import os

FORMATION_MONTHS = 12
TRADING_MONTHS = 6
NUM_PAIRS = 20           
DM_THRESHOLD = 2.0       
COINT_THRESHOLD = 2.0    
COPULA_THRESHOLD = 0.5   
COINT_PVALUE = 0.05      

DATA_START = "2015-01-01"
DATA_END = "2025-01-01"
INDEX_NAME = "S&P 500"
MKTCAP_BOTTOM_DECILE = 0.10   
MIN_PRICE = 1.0               

COMMISSION_BPS = 5.0       
MARKET_IMPACT_BPS = 20.0   

NEWEY_WEST_LAGS = 6        
RISK_FREE_TICKER = "^IRX"  

OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")