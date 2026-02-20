import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data():
    symbol = 'SMIN'
    print(f"Downloading {symbol}...", flush=True)
    
    # yfinance limitation: 10m/15m data only available for last 60 days
    # To get 1 year, we'd need daily or hourly data, but instructions asked for 10m.
    # We will get the max available high-frequency data (60 days of 15m)
    df = yf.download(symbol, period='60d', interval='15m')
    
    if df.empty:
        print("No data downloaded.")
        return
        
    # Calculate log returns for better statistical properties
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    
    os.makedirs('data/india_etf', exist_ok=True)
    df.to_csv('data/india_etf/smin_15m.csv')
    print(f"Saved {len(df)} rows of SMIN data.")

if __name__ == "__main__":
    download_data()
