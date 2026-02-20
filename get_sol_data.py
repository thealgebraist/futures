import requests
import pandas as pd
import time
import os

def fetch_binance_1m(symbol, start_time_ms, end_time_ms):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    current_start = start_time_ms
    
    while current_start < end_time_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
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
            current_start = batch[-1][0] + 60000 
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
    end_time = int(time.time() * 1000)
    start_time = end_time - (90 * 24 * 60 * 60 * 1000)
    
    print("Downloading 3 months of SOL 1m data from Binance...")
    sol_df = fetch_binance_1m("SOLUSDT", start_time, end_time)
    
    os.makedirs('data/sol_experiment', exist_ok=True)
    sol_df.to_csv('data/sol_experiment/sol_1m_3mo.csv')
    print(f"\nSaved {len(sol_df)} rows of SOL data.")

if __name__ == "__main__":
    main()
