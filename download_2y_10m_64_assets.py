import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta

def download_futures_10m(symbol, days=730): # Changed to 730 days for 2 years
    url = "https://fapi.binance.com/fapi/v1/klines"
    total_expected = days * 24 * 12 # 5m interval
    
    all_data = []
    end_time = int(time.time() * 1000)
    
    print(f"Downloading {days} days of {symbol} 5m data (resampling to 10m)...")
    
    while len(all_data) < total_expected:
        params = {
            'symbol': symbol,
            'interval': '5m',
            'limit': 1500,
            'endTime': end_time
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                print(f"Error {symbol}: {r.status_code} - {r.text}")
                break
            data = r.json()
            if not data or len(data) == 0:
                break
            
            all_data = data + all_data
            end_time = data[0][0] - 1
            
            # Significantly increased sleep time to respect Binance API limits
            time.sleep(2) # 2 seconds between requests for robustness
            
            earliest_time = data[0][0]
            if earliest_time < (int(time.time() * 1000) - (days * 24 * 60 * 60 * 1000)):
                break

        except Exception as e:
            print(f"Exception {symbol}: {e}")
            break
            
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 
        'taker_quote_vol', 'ignore'
    ])
    
    df['close'] = pd.to_numeric(df['close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    # Resample to 10m
    df_10m = df['close'].resample('10min').last().ffill()
    
    os.makedirs('data/returns_2y', exist_ok=True) # New output directory
    out_path = f'data/returns_2y/{symbol}_10m.csv'
    df_10m.to_csv(out_path)
    print(f"Saved {len(df_10m)} rows to {out_path}")
    return len(df_10m)

def main():
    assets = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT',
        'TRXUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'LTCUSDT', 'UNIUSDT', 'BCHUSDT',
        'NEARUSDT', 'FETUSDT', 'APTUSDT', 'ICPUSDT', 'STXUSDT', 'OPUSDT', 'FILUSDT',
        'XLMUSDT', 'AAVEUSDT', 'GRTUSDT', 'HBARUSDT', 'KASUSDT', 'SUIUSDT', 'ARBUSDT',
        'RENDERUSDT', 'PEPEUSDT', 'WLDUSDT', 'IMXUSDT', 'INJUSDT', 'TIAUSDT', 'LDOUSDT',
        'MKRUSDT', 'KNCUSDT', 'CRVUSDT', 'MANAUSDT', 'EGLDUSDT', 'ENJUSDT', 'CHZUSDT',
        'ZILUSDT', 'SNXUSDT', 'HOTUSDT', 'DYDXUSDT', 'FLOWUSDT', 'IOSTUSDT', 'IOTAUSDT',
        'QTUMUSDT', 'RAYUSDT', 'SXPUSDT', 'THETAUSDT', 'VETUSDT', 'SCUSDT', 'ONDOUSDT',
        'ONEUSDT', 'ONTUSDT', 'SUSHIUSDT', 'ALGOUSDT', 'DGBUSDT', 'ALPHAUSDT', 'ANKRUSDT',
        'GLMUSDT'
    ] # Total 64 assets
    
    # Check existing data and prioritize
    existing_assets_2y = [f.replace('_10m_2y.csv', '') for f in os.listdir('data/ecoins_2y') if f.endswith('_10m_2y.csv')]
    
    for ticker in assets:
        if ticker in existing_assets_2y:
            print(f"Skipping {ticker}, 2 years data already exists in data/ecoins_2y.")
            continue
        download_futures_10m(ticker)

if __name__ == "__main__":
    main()
