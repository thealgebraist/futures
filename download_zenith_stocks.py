import yfinance as yf
import pandas as pd
import os

def download_stocks():
    # 32 Top Common Stocks
    symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
        'UNH', 'JPM', 'JNJ', 'V', 'XOM', 'MA', 'AVGO', 'HD',
        'PG', 'COST', 'ABBV', 'ADBE', 'CRM', 'KO', 'PEP', 'CVX',
        'MRK', 'TMO', 'WMT', 'MCD', 'BAC', 'ACN', 'CSCO', 'ORCL'
    ]
    
    os.makedirs('data/zenith_stocks', exist_ok=True)
    
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        # 15m data is available for 60 days. 
        # To get 1 year of high-frequency data, we'd need a paid API.
        # However, for this experiment, I will use 60 days of 15m data 
        # as it's the max available for free via yfinance at this interval.
        df = yf.download(symbol, period='60d', interval='15m')
        if not df.empty:
            df.to_csv(f'data/zenith_stocks/{symbol}_15m.csv')
        else:
            print(f"Failed to download {symbol}")

if __name__ == "__main__":
    download_stocks()
