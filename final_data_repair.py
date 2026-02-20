import requests
import pandas as pd
import os
import time

def fetch_klines(symbol, interval='5m', limit=1000, start_time=None):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if start_time: params['startTime'] = int(start_time)
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200: return response.json()
    except: pass
    return []

def get_6mo_data(symbol):
    dir_path = f"data/audit/{symbol}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/10m.csv"
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - (180 * 24 * 60 * 60 * 1000)
    all_data = []
    current_start = start_time_ms
    print(f"Repairing {symbol}...", flush=True)
    while current_start < end_time_ms:
        klines = fetch_klines(symbol, start_time=current_start)
        if not klines: break
        all_data.extend(klines)
        current_start = klines[-1][6] + 1
        time.sleep(0.1)
    if all_data:
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qv', 'nt', 'tbv', 'tqv', 'i'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        df.set_index('timestamp', inplace=True)
        resampled = df.resample('10min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        resampled.to_csv(file_path)
        print(f"Repaired {len(resampled)} rows for {symbol}")

if __name__ == "__main__":
    to_fix = ['PEPEUSDT', 'BONKUSDT', 'SHIBUSDT', 'SUSHIUSDT', 'ONDOUSDT', 'TRXUSDT', 'WLDUSDT', 'XTZUSDT', 'TONUSDT', 'ATOMUSDT', 'HYPEUSDT']
    for ticker in to_fix:
        get_6mo_data(ticker)
