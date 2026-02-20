import pandas as pd
import requests
import time
import os

def try_bitget(symbol='LCAI_USDT', period='1h', days=30):
    # Bitget V1
    url = "https://api.bitget.com/api/spot/v1/market/candles"
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 3600 * 1000)
    
    params = {
        'symbol': symbol,
        'period': period,
        'after': start_time,
        'before': end_time,
        'limit': 200
    }
    try:
        r = requests.get(url, params=params)
        res = r.json()
        if res.get('code') == '00000':
            data = res.get('data', [])
            if data:
                print(f"Bitget V1 Success: {len(data)} rows")
                return data, 'bitget_v1'
        print(f"Bitget V1 Failed: {res}")
    except Exception as e:
        print(f"Bitget V1 Error: {e}")
        
    # Bitget V2
    url = "https://api.bitget.com/api/v2/spot/market/history-candles"
    params = {
        'symbol': symbol,
        'granularity': period,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 200
    }
    try:
        r = requests.get(url, params=params)
        res = r.json()
        if res.get('code') == '00000':
            data = res.get('data', [])
            if data:
                print(f"Bitget V2 Success: {len(data)} rows")
                return data, 'bitget_v2'
        print(f"Bitget V2 Failed: {res}")
    except Exception as e:
        print(f"Bitget V2 Error: {e}")
        
    return None, None

if __name__ == "__main__":
    data, source = try_bitget()
    if data:
        print(f"Source: {source}, Sample: {data[0]}")
    else:
        # Try without underscore
        data, source = try_bitget(symbol='LCAIUSDT')
        if data:
            print(f"Source: {source}, Sample: {data[0]}")
