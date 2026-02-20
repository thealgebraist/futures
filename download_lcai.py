import pandas as pd
import requests
import time
import os

def download_bitmart_hourly(symbol='LCAI_USDT', days=365):
    url = "https://api-cloud.bitmart.com/spot/quotation/v3/klines"
    # Step 60 for hourly data
    step = 60
    
    # Start from current time and go backwards
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 3600)
    
    all_data = []
    
    print(f"Downloading {days} days of {symbol} hourly data...")
    
    current_end = end_time
    while current_end > start_time:
        # Request in chunks
        params = {
            'symbol': symbol,
            'step': step,
            'before': current_end,
            'limit': 200
        }
        try:
            r = requests.get(url, params=params)
            res = r.json()
            if res.get('code') != 1000:
                print(f"API Error: {res}")
                break
            
            data = res.get('data', [])
            if not data:
                break
            
            # BitMart V3 returns list of lists: [timestamp, open, high, low, close, volume, qv]
            # Data is usually sorted in reverse (newest first)
            all_data = data + all_data
            
            # Update current_end to the timestamp of the last element (oldest in this batch)
            current_end = int(data[-1][0]) - 1
            
            print(f"Downloaded {len(all_data)} rows. Current date: {pd.to_datetime(current_end, unit='s')}", flush=True)
            
            if len(data) < 200:
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    if not all_data:
        print("No data downloaded.")
        return

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'qv'
    ])
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
        
    df['Datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[['Datetime', 'close']]
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/lcai_hourly.csv', index=False)
    print(f"Saved {len(df)} rows to data/lcai_hourly.csv")

if __name__ == "__main__":
    download_bitmart_hourly()
