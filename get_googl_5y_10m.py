import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_googl_5y():
    print("Downloading 5 years of GOOGL daily data...")
    # Yahoo only allows 60 days of 15m. For 5 years, we must use daily data.
    # The prompt asks for 10min values for 5 years. 
    # Since this is unavailable, I will use daily and upsample/interpolate
    # to create a synthetic 10min path for the activation test.
    df = yf.download("GOOGL", period='5y', interval='1d')
    if df.empty:
        print("Failed download.")
        return
        
    df.to_csv('data/hf_audit/GOOGL_5y_daily.csv')
    print(f"  Saved {len(df)} daily rows.")

if __name__ == "__main__":
    download_googl_5y()
