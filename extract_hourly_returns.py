import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def extract_hourly_returns():
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
    ]

    all_hourly_returns = []

    for ticker in assets:
        path = ""
        data_source_type = "" # To distinguish between formats

        if os.path.exists(f"data/returns_2y/{ticker}_10m.csv"):
            path = f"data/returns_2y/{ticker}_10m.csv"
            data_source_type = "returns_2y"
        elif os.path.exists(f"data/ecoins_2y/{ticker}_10m_2y.csv"):
            path = f"data/ecoins_2y/{ticker}_10m_2y.csv"
            data_source_type = "ecoins_2y"
        elif os.path.exists(f"data/audit/{ticker}/10m.csv"):
            path = f"data/audit/{ticker}/10m.csv"
            data_source_type = "audit"
        
        if not path:
            print(f"Warning: Data not found for {ticker}. Skipping hourly analysis for this asset.")
            continue
        
        print(f"Processing hourly returns for {ticker} from {path} (type: {data_source_type})...")
        try:
            df = None
            if data_source_type == "returns_2y":
                # These files have header 'timestamp,close'
                df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
                df['close'] = pd.to_numeric(df['close'], errors='coerce') # Ensure numeric
            elif data_source_type == "ecoins_2y":
                # These files have header 'Open_time,Close,Volume'
                df = pd.read_csv(path, index_col='Open_time', parse_dates=True)
                df = df[['Close']].rename(columns={'Close': 'close'})
                df['close'] = pd.to_numeric(df['close'], errors='coerce') # Ensure numeric
            elif data_source_type == "audit":
                # These files have header 'timestamp,open,high,low,close,volume'
                df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
                df = df[['close']] # Select the 'close' column directly
                df['close'] = pd.to_numeric(df['close'], errors='coerce') # Ensure numeric
            
            if df is None or df.empty:
                print(f"  Error: DataFrame is empty for {ticker} from {path}. Skipping.")
                continue

            # Ensure index is datetime type for consistent hour extraction
            df.index = pd.to_datetime(df.index)

            # Calculate log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df = df.dropna()

            # Extract hour of day
            df['hour_of_day'] = df.index.hour

            # Append to overall list
            for index, row in df.iterrows():
                if np.isfinite(row['log_return']):
                    all_hourly_returns.append({'hour_of_day': row['hour_of_day'], 'return_value': row['log_return']})

        except Exception as e:
            print(f"Error processing {ticker} from {path}: {e}")
            continue

    if not all_hourly_returns:
        print("No hourly returns data collected.")
        return

    hourly_returns_df = pd.DataFrame(all_hourly_returns)
    os.makedirs('data', exist_ok=True)
    hourly_returns_df.to_csv('data/hourly_returns.csv', index=False)
    print(f"Saved {len(hourly_returns_df)} hourly return samples to data/hourly_returns.csv")

if __name__ == "__main__":
    extract_hourly_returns()
