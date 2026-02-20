import pandas as pd
import requests
import time
import os

def download_binance_data(symbol, interval, months=6):
    url = "https://api.binance.com/api/v3/klines"
    
    # Map months to minutes
    if interval == '1m':
        total_rows = months * 30 * 24 * 60
        api_interval = '1m'
    elif interval == '10m':
        # Binance doesn't have 10m directly, we get 5m or 15m and resample OR just use 10m logic in preparation
        # Actually Binance has 1m, 3m, 5m, 15m, 30m, 1h, etc.
        # We will download 5m and resample to 10m.
        total_rows = months * 30 * 24 * 12 * 2 # twice the 10m rows
        api_interval = '5m'
    elif interval == '1h':
        total_rows = months * 30 * 24
        api_interval = '1h'
    else:
        return
        
    all_data = []
    end_time = int(time.time() * 1000)
    
    print(f"Downloading {symbol} {interval} for {months} months...")
    
    rows_collected = 0
    while rows_collected < total_rows:
        params = {
            'symbol': symbol,
            'interval': api_interval,
            'limit': 1000,
            'endTime': end_time
        }
        try:
            r = requests.get(url, params=params)
            data = r.json()
            if not data or len(data) == 0:
                break
            
            all_data = data + all_data
            rows_collected += len(data)
            end_time = data[0][0] - 1
            
            if rows_collected % 5000 == 0:
                print(f"  {symbol} {interval}: Collected {rows_collected}/{total_rows} rows...")
            
            time.sleep(0.05) # fast but safe
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 
        'taker_quote_vol', 'ignore'
    ])
    
    # Convert and resample if needed
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = pd.to_numeric(df['close'])
    df = df.set_index('timestamp')
    
    if interval == '10m':
        df = df['close'].resample('10min').last().ffill()
    else:
        df = df['close']
        
    os.makedirs(f'data/audit/{symbol}', exist_ok=True)
    filename = f'data/audit/{symbol}/{interval}.csv'
    df.to_csv(filename)
    print(f"  Saved to {filename}")

if __name__ == "__main__":
    coins = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 
        'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'SHIBUSDT', 'DOTUSDT', 
        'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'PEPEUSDT', 'WIFUSDT', 'RENDERUSDT'
    ]
    intervals = ['1m', '10m', '1h']
    
    for coin in coins:
        for interval in intervals:
            download_binance_data(coin, interval)
