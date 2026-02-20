import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data():
    # 8 top/micro symbols
    symbols = ['ES=F', 'NQ=F', 'RTY=F', 'CL=F', 'GC=F', 'MES=F', 'MNQ=F', 'M2K=F']
    data_frames = {}
    
    for symbol in symbols:
        print(f"Downloading {symbol}...", flush=True)
        df = yf.download(symbol, period='60d', interval='5m')
        if not df.empty:
            resampled = df['Close'].resample('10min').last()
            data_frames[symbol] = resampled
            
    if not data_frames:
        print("No data downloaded.", flush=True)
        return
        
    combined = pd.concat(data_frames, axis=1)
    combined.columns = symbols
    combined = combined.ffill().dropna()
    
    os.makedirs('data', exist_ok=True)
    combined.to_csv('data/futures_10m_v2.csv')
    print(f"Saved {len(combined)} rows of 8-feature resampled 10m data.", flush=True)

if __name__ == "__main__":
    download_data()
