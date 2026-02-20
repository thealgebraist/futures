import pandas as pd
import requests
import time
import os

def try_mexc(symbol='LCAIUSDT', interval='1h', limit=500):
    url = "https://api.mexc.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    try:
        r = requests.get(url, params=params)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"MEXC Success: {len(data)} rows")
            return data, 'mexc'
        print(f"MEXC Failed: {data}")
    except Exception as e:
        print(f"MEXC Error: {e}")
    return None, None

if __name__ == "__main__":
    data, source = try_mexc()
    if data:
        print(f"Source: {source}, Sample: {data[0]}")
    else:
        # Try LCAI_USDT
        data, source = try_mexc(symbol='LCAI_USDT')
        if data:
            print(f"Source: {source}, Sample: {data[0]}")
