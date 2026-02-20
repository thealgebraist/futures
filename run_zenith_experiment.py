import os
import subprocess
import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt

# --- GARCH Filter ---
def estimate_garch(returns, omega=1e-6, alpha=0.1, beta=0.8):
    n = len(returns)
    sigmas = np.zeros(n)
    sigmas[0] = np.max([np.var(returns), 1e-6])
    for t in range(1, n):
        sigmas[t] = omega + alpha * (returns[t-1]**2) + beta * sigmas[t-1]
    return np.sqrt(sigmas)

class ZenithInference:
    def __init__(self, model_path):
        self.input_dim = 16
        self.hidden_dim = 512
        self.output_dim = 4
        
        with open(model_path, 'rb') as f:
            self.W1 = np.frombuffer(f.read(self.input_dim * self.hidden_dim * 4), dtype=np.float32).reshape(self.input_dim, self.hidden_dim)
            self.b1 = np.frombuffer(f.read(self.hidden_dim * 4), dtype=np.float32)
            self.W2 = np.frombuffer(f.read(self.hidden_dim * self.hidden_dim * 4), dtype=np.float32).reshape(self.hidden_dim, self.hidden_dim)
            self.b2 = np.frombuffer(f.read(self.hidden_dim * 4), dtype=np.float32)
            self.W_out = np.frombuffer(f.read(self.hidden_dim * self.output_dim * 4), dtype=np.float32).reshape(self.hidden_dim, self.output_dim)
            self.b_out = np.frombuffer(f.read(self.output_dim * 4), dtype=np.float32)

    def forward(self, x):
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        y = np.dot(h2, self.W_out) + self.b_out
        return y

def run_experiment():
    symbols = [f.split('_')[0] for f in os.listdir('data/zenith_stocks') if f.endswith('.csv')]
    os.makedirs('models/zenith_stocks', exist_ok=True)
    
    results = []
    
    for symbol in symbols:
        csv_path = f'data/zenith_stocks/{symbol}_15m.csv'
        model_path = f'models/zenith_stocks/{symbol}.bin'
        
        print(f"\n--- Processing {symbol} ---", flush=True)
        # 1. Train
        subprocess.run(['./zenith_trainer', csv_path, model_path], check=True)
        
        # 2. Backtest
        df = pd.read_csv(csv_path)
        prices = pd.to_numeric(df['Close'], errors='coerce').values
        prices = prices[~np.isnan(prices)]
        
        train_split = int(len(prices) * 0.8)
        test_prices = prices[train_split:]
        
        # Stats for normalization
        mean = np.mean(prices[:train_split])
        std = np.std(prices[:train_split])
        p_scaled = (prices - mean) / (std + 1e-6)
        
        # GARCH for Gating
        hist_rets = np.diff(prices[:train_split]) / prices[:train_split-1]
        hist_vols = estimate_garch(hist_rets)
        vol_limit = 1.5 * np.mean(hist_vols)
        
        test_rets = np.diff(test_prices) / test_prices[:-1]
        test_vols = estimate_garch(test_rets)
        
        # Inference
        model = ZenithInference(model_path)
        
        capital = 100.0
        equity = [capital]
        input_steps = 16
        
        for i in range(len(test_prices) - input_steps - 1):
            x_window = p_scaled[train_split + i : train_split + i + input_steps]
            pred = model.forward(x_window)
            
            pred_change = np.mean(pred) - x_window[-1]
            current_vol = test_vols[i + input_steps - 1] if (i + input_steps - 1) < len(test_vols) else 0
            
            pos = 0
            if current_vol < vol_limit:
                if pred_change < -0.0001: # Short Only
                    pos = -1
            
            actual_ret = (test_prices[i + input_steps] - test_prices[i + input_steps - 1]) / test_prices[i + input_steps - 1]
            capital *= (1 + actual_ret * pos)
            equity.append(capital)
            
        results.append({
            'symbol': symbol,
            'final_equity': capital,
            'max_dd': (np.max(equity) - np.min(equity)) / np.max(equity)
        })
        print(f"  {symbol} Final: ${capital:.2f}", flush=True)

    pd.DataFrame(results).to_csv('zenith_stocks_experiment_results.csv', index=False)

if __name__ == "__main__":
    run_experiment()
