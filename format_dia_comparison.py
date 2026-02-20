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
                    data[symbol] = float(parts[-1])
    except: pass
    return data

def main():
    ffnn_orig = load_res('res_ffnn256.txt')
    dia_new = load_res('res_dia_new.txt')
    
    # Symbols to compare
    s = 'DIA'
    
    # We need current prices to normalize
    df = pd.read_csv(f'data/full_history/{s}_max.csv')
    curr = float(df['Close'].iloc[-1])
        
    print(f"Comparison for {s} ($100 base):")
    print(f"FFNN 256 (5min):  ${(ffnn_orig.get(s, 0) / curr * 100):.2f}")
    print(f"FFNN 512 (10min): ${(dia_new.get('DIA_FFNN_512', 0) / curr * 100):.2f}")
    print(f"FFNN 64-64 (10min): ${(dia_new.get('DIA_FFNN_64_64', 0) / curr * 100):.2f}")

if __name__ == "__main__":
    main()
