import yfinance as yf
import requests
import pandas as pd
import numpy as np
import time
import os

def download_googl():
    print("Downloading GOOGL 15m data...")
    df = yf.download("GOOGL", period='60d', interval='15m')
    if not df.empty:
        os.makedirs('data/hf_audit', exist_ok=True)
        df.to_csv('data/hf_audit/GOOGL_15m.csv')
        print(f"  GOOGL: {len(df)} rows.")

def fetch_binance_5m(symbol, start_time_ms, end_time_ms):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    current_start = start_time_ms
    while current_start < end_time_ms:
        params = {"symbol": symbol, "interval": "5m", "limit": 1000, "startTime": current_start}
        r = requests.get(url, params=params)
        if r.status_code != 200: break
        batch = r.json()
        if not batch: break
        data.extend(batch)
        current_start = batch[-1][0] + 300000
        time.sleep(0.02)
    df = pd.DataFrame(data, columns=['OT','O','H','L','C','V','CT','QV','NT','TB','TQ','I'])
    df['OT'] = pd.to_datetime(df['OT'], unit='ms')
    df = df.set_index('OT')
    return df[['C']].astype(float)

def download_sol():
    print("Downloading SOLUSDT 5m data (for 10m resampling)...")
    end_time = int(time.time() * 1000)
    start_time = end_time - (60 * 24 * 60 * 60 * 1000) # 60 days
    df = fetch_binance_5m("SOLUSDT", start_time, end_time)
    if not df.empty:
        os.makedirs('data/hf_audit', exist_ok=True)
        df.to_csv('data/hf_audit/SOLUSDT_5m.csv')
        print(f"  SOLUSDT: {len(df)} rows.")

if __name__ == "__main__":
    download_googl()
    download_sol()
