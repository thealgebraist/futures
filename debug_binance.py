import requests
import time

url = "https://fapi.binance.com/fapi/v1/klines"
symbol = "SUIUSDT"
params = {
    'symbol': symbol,
    'interval': '10m',
    'limit': 1000,
    'startTime': int(time.time() * 1000) - (24 * 60 * 60 * 1000) # 1 day ago
}

response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
