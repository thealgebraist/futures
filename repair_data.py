import requests
import pandas as pd
import os
import time

def fetch_klines(symbol, interval='5m', limit=1000, start_time=None):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if start_time:
        params['startTime'] = int(start_time)
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {symbol}: {response.status_code}")
        return []

def repair_data(symbol):
    print(f"Repairing {symbol}...")
    dir_path = f"data/audit/{symbol}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/10m.csv"
    
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - (180 * 24 * 60 * 60 * 1000)
    
    all_data = []
    current_start = start_time_ms
    while current_start < end_time_ms:
        klines = fetch_klines(symbol, start_time=current_start)
        if not klines:
            break
        all_data.extend(klines)
        current_start = klines[-1][6] + 1
        time.sleep(0.05)
    
    if all_data:
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'ct', 'qav', 'nt', 'tbba', 'tbqa', 'i'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df.set_index('timestamp', inplace=True)
        resampled = df.resample('10min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        resampled.to_csv(file_path)
        print(f"Fixed {symbol}: {len(resampled)} rows.")
    else:
        print(f"Failed to fetch data for {symbol}")

missing_assets = [
    'ADAUSDT', 'AVAXUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 
    'DOTUSDT', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 
    'PEPEUSDT', 'RENDERUSDT', 'SHIBUSDT', 'SOLUSDT', 'WIFUSDT', 'XRPUSDT'
]

for asset in missing_assets:
    repair_data(asset)
