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
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"Error: {r.status_code}")
            break
        
        batch = r.json()
        if not batch:
            break
            
        data.extend(batch)
        current_start = batch[-1][0] + 60000 
        print(f"Fetched up to {pd.to_datetime(current_start, unit='ms')}", end='\r')
        time.sleep(0.1) 
        
    df = pd.DataFrame(data, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df = df.set_index('Open_time')
    return df[['Close']].astype(float)

def main():
    end_time = int(time.time() * 1000)
    start_time = end_time - (90 * 24 * 60 * 60 * 1000)
    
    print("Downloading 3 months of BTC 1m data from Binance...")
    btc_df = fetch_binance_1m("BTCUSDT", start_time, end_time)
    
    os.makedirs('data/btc_experiment', exist_ok=True)
    btc_df.to_csv('data/btc_experiment/btc_1m_3mo.csv')
    print(f"\nSaved {len(btc_df)} rows of BTC data.")

if __name__ == "__main__":
    main()
