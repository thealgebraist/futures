import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data():
    # 16 Diverse Futures
    symbols = [
        'ES=F', 'NQ=F', 'RTY=F', 'YM=F',  # Indices
        'CL=F', 'NG=F', 'RB=F', 'HO=F',  # Energy
        'GC=F', 'SI=F', 'HG=F', 'PL=F',  # Metals
        '6E=F', '6J=F', 'ZB=F', 'ZC=F'   # FX, Bonds, Ag
    ]
    
    data_frames = {}
    for symbol in symbols:
        print(f"Downloading {symbol}...", flush=True)
        # 5m data is available for 60 days.
        df = yf.download(symbol, period='60d', interval='5m')
        if not df.empty:
            # Resample to 10m
            resampled = df['Close'].resample('10min').last()
            # Calculate price changes (returns)
            data_frames[symbol] = resampled.pct_change()
            
    if not data_frames:
        print("No data downloaded.")
        return
        
    combined = pd.concat(data_frames, axis=1)
    combined.columns = symbols
    combined = combined.dropna() # Drops rows with any NaNs (including first row from pct_change)
    
    os.makedirs('data', exist_ok=True)
    combined.to_csv('data/futures_16_changes.csv')
    print(f"Saved {len(combined)} rows of 16-future changes.")

if __name__ == "__main__":
    download_data()
