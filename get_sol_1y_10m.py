import requests
import pandas as pd
import time
import os

def fetch_binance_interval(symbol, interval, start_time_ms, end_time_ms):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    current_start = start_time_ms
    
    # Map interval to ms
    interval_ms = {
        "1m": 60000,
        "5m": 300000,
        "15m": 900000,
        "1h": 3600000
    }[interval]
    
    while current_start < end_time_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "startTime": current_start
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                print(f"Error: {r.status_code}")
                break
            batch = r.json()
            if not batch:
                break
            data.extend(batch)
            current_start = batch[-1][0] + interval_ms
            print(f"Fetched up to {pd.to_datetime(current_start, unit='ms')}", end='\r', flush=True)
            time.sleep(0.05) 
        except Exception as e:
            print(f"Exception: {e}")
            break
        
    df = pd.DataFrame(data, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df = df.set_index('Open_time')
    return df[['Close', 'Volume']].astype(float)

def main():
    # 1 year = 365 days
    end_time = int(time.time() * 1000)
    start_time = end_time - (365 * 24 * 60 * 60 * 1000)
    
    print("Downloading 1 year of SOL 5m data from Binance...")
    # Using 5m to resample to 10m exactly
    sol_df = fetch_binance_interval("SOLUSDT", "5m", start_time, end_time)
    
    if sol_df.empty:
        print("Failed to download data.")
        return

    # Resample to 10m
    sol_10m = sol_df['Close'].resample('10min').last().ffill()
    sol_vol_10m = sol_df['Volume'].resample('10min').sum()
    
    final_df = pd.concat([sol_10m, sol_vol_10m], axis=1)
    
    os.makedirs('data/sol_experiment', exist_ok=True)
    final_df.to_csv('data/sol_experiment/sol_10m_1y.csv')
    print(f"\nSaved {len(final_df)} rows of resampled 10m SOL data.")

if __name__ == "__main__":
    main()
