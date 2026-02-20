import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data():
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
    os.makedirs('data/signature_experiment', exist_ok=True)
    
    for symbol in symbols:
        print(f"Downloading 4y data for {symbol}...")
        df = yf.download(symbol, period='4y', interval='1d')
        if not df.empty:
            df.to_csv(f'data/signature_experiment/{symbol}_4y.csv')
            print(f"  {symbol}: {len(df)} rows.")

if __name__ == "__main__":
    download_data()
