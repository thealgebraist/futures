import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data():
    symbols = ['EURUSD=X', 'CORN', 'WEAT', 'DBA']
    data_frames = {}
    
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        # 15m data is available for 60 days.
        df = yf.download(symbol, period='60d', interval='15m')
        if not df.empty:
            # Price changes (returns)
            data_frames[symbol] = df['Close'].pct_change()
            
    if not data_frames:
        print("No data downloaded.")
        return
        
    combined = pd.concat(data_frames, axis=1)
    combined.columns = symbols
    combined = combined.dropna()
    
    os.makedirs('data/commodities', exist_ok=True)
    combined.to_csv('data/commodities/returns_15m.csv')
    print(f"Saved {len(combined)} rows of commodity/fx returns (15m).")

if __name__ == "__main__":
    download_data()
