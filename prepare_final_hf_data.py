import pandas as pd
import numpy as np
import os

def prepare_data():
    os.makedirs('data/final_hf', exist_ok=True)
    
    # 1. GOOGL
    googl_path = 'data/hf_audit/GOOGL_15m.csv'
    if os.path.exists(googl_path):
        # Yahoo hf format has extra header rows (Ticker, Date etc)
        # Let's try reading with standard header then dropping if non-date
        df = pd.read_csv(googl_path)
        # Identify start of data
        # Data rows start with dates like 2025-XX-XX
        df['dt'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.dropna(subset=['dt'])
        df = df.set_index('dt')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df_10m = df['Close'].resample('10min').last().ffill()
        df_10m.to_csv('data/final_hf/GOOGL_10m.csv')
        print(f"Prepared GOOGL 10m: {len(df_10m)} rows")

    # 2. SOLUSDT
    sol_path = 'data/hf_audit/SOLUSDT_5m.csv'
    if os.path.exists(sol_path):
        df = pd.read_csv(sol_path)
        df['dt'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.dropna(subset=['dt'])
        df = df.set_index('dt')
        col = 'C' if 'C' in df.columns else 'Close'
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df_10m = df[col].resample('10min').last().ffill()
        df_10m.to_csv('data/final_hf/SOLUSDT_10m.csv')
        print(f"Prepared SOLUSDT 10m: {len(df_10m)} rows")

if __name__ == "__main__":
    prepare_data()
