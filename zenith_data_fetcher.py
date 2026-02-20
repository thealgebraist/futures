import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_klines(symbol, interval='5m', limit=1000, start_time=None):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time:
        params['startTime'] = int(start_time)
        
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {symbol}: {response.status_code}")
        return []

def get_6mo_data(symbol):
    dir_path = f"data/audit/{symbol}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/10m.csv"
    
    # Approx 6 months ago in ms
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - (180 * 24 * 60 * 60 * 1000) # 180 days
    
    all_data = []
    current_start = start_time_ms
    
    print(f"Fetching {symbol} (5m -> 10m)...")
    while current_start < end_time_ms:
        klines = fetch_klines(symbol, interval='5m', start_time=current_start)
        if not klines:
            break
        
        all_data.extend(klines)
        current_start = klines[-1][6] + 1
        time.sleep(0.05)
    
    if all_data:
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        
        # Resample 5m to 10m
        df.set_index('timestamp', inplace=True)
        resampled = df.resample('10min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled.to_csv(file_path)
        print(f"Saved {len(resampled)} rows for {symbol} to {file_path}")
    else:
        print(f"No data found for {symbol}")

if __name__ == "__main__":
    MID_32 = [
        'LINKUSDT', 'POLUSDT', 'UNIUSDT', 'NEARUSDT', 'LDOUSDT', 
        'ICPUSDT', 'FILUSDT', 'ARBUSDT', 'OPUSDT', 'TIAUSDT', 
        'RENDERUSDT', 'FETUSDT', 'GRTUSDT', 'AAVEUSDT', 'CRVUSDT', 
        'SNXUSDT', 'MKRUSDT', 'DYDXUSDT', 'IMXUSDT', 'STXUSDT', 
        'KASUSDT', 'INJUSDT', 'SEIUSDT', 'SUIUSDT', 'APTUSDT', 
        'ARUSDT', 'THETAUSDT', 'FLOWUSDT', 'EGLDUSDT', 'ALGOUSDT', 
        'HBARUSDT', 'VETUSDT'
    ]
    LOW_32 = [
        'ANKRUSDT', 'CHZUSDT', 'ENJUSDT', 'BATUSDT', 'KAVAUSDT', 
        'ZILUSDT', 'RVNUSDT', 'SCUSDT', 'HOTUSDT', 'ONEUSDT', 
        'IOTAUSDT', 'ONTUSDT', 'QTUMUSDT', 'IOSTUSDT', 'ZRXUSDT', 
        'OMGUSDT', 'GLMUSDT', 'SXPUSDT', 'ALPHAUSDT', 'AUDIOUSDT', 
        'BANDUSDT', 'COTIUSDT', 'DGBUSDT', 'FUNUSDT', 'KNCUSDT', 
        'LRCUSDT', 'MTLUSDT', 'NMRUSDT', 'OGNUSDT', 'RAYUSDT', 
        'REQUSDT', 'STORJUSDT'
    ]
    
    all_tickers = MID_32 + LOW_32
    for ticker in all_tickers:
        # Check if 10m.csv is already enough (approx 26000 rows)
        fpath = f"data/audit/{ticker}/10m.csv"
        if os.path.exists(fpath):
            try:
                # Optimized check: only count rows
                count = sum(1 for line in open(fpath))
                if count > 24000:
                    print(f"Skipping {ticker}, already has {count} rows.")
                    continue
            except:
                pass
                
        get_6mo_data(ticker)
