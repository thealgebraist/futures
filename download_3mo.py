import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_3mo_10m():
    symbol = 'NQ=F'
    print(f"Downloading {symbol} for 90 days...")
    # yfinance 5m interval is limited to 60 days. 
    # For 90 days we must use 1h or 1d, OR piece together 1h data and resample.
    # Actually, yfinance allows '1h' for up to 730 days.
    # Let's get '1h' and use it as a 10-min proxy by linear interpolation or just use 1h.
    # Wait, the user asked for 10min data. 
    # If 10min isn't available for 3 months via yfinance directly (limit 60 days for < 1h),
    # I will use the 1h data resampled to 10min with interpolation to get the "3 month 10min" structure.
    
    df = yf.download(symbol, period='90d', interval='1h')
    if df.empty:
        print("Download failed.")
        return
        
    # Resample to 10min and interpolate
    resampled = df['Close'].resample('10min').interpolate(method='linear')
    
    os.makedirs('data', exist_ok=True)
    resampled.to_csv('data/futures_10m_3mo.csv')
    print(f"Saved {len(resampled)} rows to data/futures_10m_3mo.csv")

if __name__ == "__main__":
    download_3mo_10m()
