import yfinance as yf
import pandas as pd
import os

def download_benchmarks():
    # Top 8 Diverse Index Funds/ETFs
    indices = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000',
        'VTI': 'Total Stock Market',
        'VXUS': 'Total International',
        'BND': 'Total Bond Market',
        'VIG': 'Dividend Appreciation'
    }
    
    os.makedirs('data/benchmarks', exist_ok=True)
    
    # 1. Daily data for 2 years (for CAGR/Sharpe)
    print("Downloading 2y daily data for benchmarks...")
    daily_df = yf.download(list(indices.keys()), period='2y', interval='1d')['Close']
    daily_df.to_csv('data/benchmarks/indices_2y_daily.csv')
    
    # 2. 15m data for 60 days (for direct direct Zenith/Ecoin window comparison)
    print("Downloading 60d 15m data for benchmarks...")
    for symbol in indices.keys():
        df = yf.download(symbol, period='60d', interval='15m')
        if not df.empty:
            df.to_csv(f'data/benchmarks/{symbol}_15m_60d.csv')

if __name__ == "__main__":
    download_benchmarks()
