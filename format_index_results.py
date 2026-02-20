import pandas as pd
import numpy as np

def load_res(path):
    data = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 11:
                    symbol = parts[0]
                    # Year 10 is the last value
                    # We want to normalize relative to current (which is implicitly first price used)
                    # Actually, the trainers output absolute prices. 
                    # To compare index funds fairly, let's just report the raw projected value
                    # and the growth factor if possible.
                    data[symbol] = float(parts[-1])
    except: pass
    return data

def main():
    ffnn = load_res('res_ffnn256.txt')
    mc = load_res('res_mc64.txt')
    gp = load_res('res_gp.txt')
    
    symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VXUS', 'BND', 'VIG']
    
    # We need current prices to normalize
    current_prices = {}
    for s in symbols:
        df = pd.read_csv(f'data/full_history/{s}_max.csv')
        current_prices[s] = float(df['Close'].iloc[-1])
        
    print("Symbol & FFNN 256 ($) & GP ($) & MC 64 ($) ")
    for s in symbols:
        f_val = ffnn.get(s, 0)
        g_val = gp.get(s, 0)
        m_val = mc.get(s, 0)
        
        # Growth factor from $100
        f_growth = (f_val / current_prices[s]) * 100 if s in current_prices else 0
        g_growth = (g_val / current_prices[s]) * 100 if s in current_prices else 0
        m_growth = (m_val / current_prices[s]) * 100 if s in current_prices else 0
        
        print(f"{s} & {f_growth:.2f} & {g_growth:.2f} & {m_growth:.2f} ")

if __name__ == "__main__":
    main()
