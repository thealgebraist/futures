import yfinance as yf
import pandas as pd
import os

def download_stocks():
    stocks = ['NVDA', 'TSLA', 'AAPL', 'MSFT']
    os.makedirs('data/audit/stocks', exist_ok=True)
    
    for ticker in stocks:
        print(f"Downloading {ticker}...")
        # Get as much 1h data as possible (yfinance allows 2y for 1h)
        # We'll use 1h for stocks since 1m/5m is too limited for deep training depth.
        data = yf.download(ticker, period="max", interval="1h")
        if data.empty:
            print(f"Failed to download {ticker}")
            continue
            
        data.index.name = 'timestamp'
        data = data[['Close']]
        data.columns = ['close']
        
        path = f'data/audit/stocks/{ticker}.csv'
        data.to_csv(path)
        print(f"  Saved {len(data)} rows to {path}")

if __name__ == "__main__":
    download_stocks()
