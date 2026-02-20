import pandas as pd
import requests
import time
import os

def download_binance_1m(symbol='BTCUSDT', months=4):
    url = "https://api.binance.com/api/v3/klines"
    # 4 months in minutes approx: 4 * 30 * 24 * 60 = 172,800
    total_minutes = months * 30 * 24 * 60
    
    all_data = []
    # Start from current time and go backwards
    end_time = int(time.time() * 1000)
    
    print(f"Downloading {total_minutes} minutes of {symbol} data...")
    
    while len(all_data) < total_minutes:
        params = {
            'symbol': symbol,
            'interval': '1m',
            'limit': 1000,
            'endTime': end_time
        }
        try:
            r = requests.get(url, params=params)
            data = r.json()
            if not data:
                break
            
            # Binance returns data in ascending order (oldest first in the batch)
            # but we are going backwards, so we prepend
            all_data = data + all_data
            
            # Set end_time to the timestamp of the first element in the batch minus 1ms
            end_time = data[0][0] - 1
            
            if len(all_data) % 10000 == 0 or len(all_data) < 10000:
                print(f"Downloaded {len(all_data)} rows...", flush=True)
            
            # Rate limit friendly
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 
        'taker_quote_vol', 'ignore'
    ])
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
        
    df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['Datetime', 'close']]
    df.columns = ['Datetime', 'NQ=F'] # Renaming to reuse previous scripts easily
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/futures_1m_4mo.csv', index=False)
    print(f"Saved {len(df)} rows to data/futures_1m_4mo.csv")

if __name__ == "__main__":
    download_binance_1m()
