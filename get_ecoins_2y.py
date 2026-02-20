import requests
import pandas as pd
import time
import os

def fetch_binance_interval(symbol, interval, start_time_ms, end_time_ms):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    current_start = start_time_ms
    interval_ms = {"5m": 300000}
    
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
                print(f"Error {symbol}: {r.status_code}")
                break
            batch = r.json()
            if not batch:
                break
            data.extend(batch)
            current_start = batch[-1][0] + interval_ms[interval]
            time.sleep(0.02) 
        except Exception as e:
            print(f"Exception {symbol}: {e}")
            break
            
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df = df.set_index('Open_time')
    df = df[['Close', 'Volume']].astype(float)
    
    # Resample to 10m
    res_close = df['Close'].resample('10min').last().ffill()
    res_vol = df['Volume'].resample('10min').sum()
    return pd.concat([res_close, res_vol], axis=1)

def main():
    symbols = ['ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT']
    end_time = int(time.time() * 1000)
    start_time = end_time - (2 * 365 * 24 * 60 * 60 * 1000)
    
    os.makedirs('data/ecoins_2y', exist_ok=True)
    
    for symbol in symbols:
        print(f"Downloading 2y of {symbol} 5m data...")
        df = fetch_binance_interval(symbol, "5m", start_time, end_time)
        if not df.empty:
            df.to_csv(f'data/ecoins_2y/{symbol}_10m_2y.csv')
            print(f"Saved {len(df)} rows for {symbol}.")
        else:
            print(f"Failed for {symbol}.")

if __name__ == "__main__":
    main()
