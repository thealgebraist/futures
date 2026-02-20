import yfinance as yf
import pandas as pd
import os

def download_full_history():
    indices = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VXUS', 'BND', 'VIG']
    os.makedirs('data/full_history', exist_ok=True)
    
    for symbol in indices:
        print(f"Downloading all-time data for {symbol}...")
        # 'max' period for all history
        df = yf.download(symbol, period='max', interval='1d')
        if not df.empty:
            df.to_csv(f'data/full_history/{symbol}_max.csv')
            print(f"  {symbol}: {len(df)} rows.")

if __name__ == "__main__":
    download_full_history()
