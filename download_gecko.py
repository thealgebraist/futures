import pandas as pd
import requests
import time
import os

def get_geckoterminal_data(token_address='0x9ca8530ca349c966fe9ef903df17a75b8a778927'):
    # Get pools for the token
    url = f"https://api.geckoterminal.com/api/v2/networks/eth/tokens/{token_address}/pools"
    headers = {'Accept': 'application/json;version=20230302'}
    
    try:
        r = requests.get(url, headers=headers)
        res = r.json()
        pools = res.get('data', [])
        if not pools:
            print("No pools found for token.")
            return None
        
        # Take the first pool (usually the most liquid)
        pool_address = pools[0]['attributes']['address']
        print(f"Found pool: {pool_address}")
        
        # Get hourly OHLCV
        ohlcv_url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{pool_address}/ohlcv/hour"
        # GeckoTerminal limit is 1000 per request
        ohlcv_params = {'limit': 1000}
        
        r = requests.get(ohlcv_url, headers=headers, params=ohlcv_params)
        ohlcv_res = r.json()
        
        ohlcv_data = ohlcv_res.get('data', {}).get('attributes', {}).get('ohlcv_list', [])
        if not ohlcv_data:
            print("No OHLCV data found.")
            return None
            
        # Format: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['Datetime', 'close']]
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/lcai_hourly.csv', index=False)
        print(f"Saved {len(df)} rows to data/lcai_hourly.csv")
        return df
        
    except Exception as e:
        print(f"GeckoTerminal Error: {e}")
        return None

if __name__ == "__main__":
    get_geckoterminal_data()
