import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# 1. GARCH(1,1) Estimation
def estimate_garch(returns, omega=1e-6, alpha=0.1, beta=0.8):
    """
    Simple recursive GARCH(1,1) volatility estimation.
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    """
    n = len(returns)
    sigmas = np.zeros(n)
    sigmas[0] = np.var(returns)
    
    for t in range(1, n):
        sigmas[t] = omega + alpha * (returns[t-1]**2) + beta * sigmas[t-1]
        
    return np.sqrt(sigmas)

# 2. Zenith Engine: Short-Only with GARCH Gating
class ZenithEngine:
    def __init__(self, model_path='predictive_ffnn.pth'):
        # Model definition (matches earlier FFNN)
        class FFNN(nn.Module):
            def __init__(self, input_dim=16, hidden_dim=512, output_dim=4):
                super(FFNN, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            def forward(self, x):
                return self.net(x)
        
        self.model = FFNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.threshold = 0.0001
        self.input_steps = 16

    def run_backtest(self, csv_path='data/futures_1m_4mo.csv'):
        df = pd.read_csv(csv_path)
        prices = df['NQ=F'].values
        prices = prices[~np.isnan(prices)]
        
        # Calculate returns for GARCH
        rets = np.diff(prices) / prices[:-1]
        volatilities = estimate_garch(rets)
        
        # Normalize prices for FFNN
        scaler = StandardScaler()
        p_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Split (use last month for backtest as before)
        split_idx = int(0.8 * len(prices))
        test_prices = prices[split_idx:]
        test_p_scaled = p_scaled[split_idx:]
        test_vols = volatilities[split_idx:]
        
        capital = 100.0
        equity = [capital]
        
        # Risk-off threshold: volatility > 1.5 * mean_vol
        vol_limit = 1.5 * np.mean(volatilities)
        
        num_steps = len(test_p_scaled) - self.input_steps - 1
        
        # Batch inference for speed
        X_all = []
        for i in range(num_steps):
            X_all.append(test_p_scaled[i : i + self.input_steps])
        X_all = torch.FloatTensor(np.array(X_all))
        
        with torch.no_grad():
            preds = self.model(X_all).numpy()
            
        # Analysis loop
        for i in range(num_steps):
            last_p_scaled = test_p_scaled[i + self.input_steps - 1]
            pred_change = np.mean(preds[i]) - last_p_scaled
            
            # GARCH-Gated Logic
            pos = 0
            current_vol = test_vols[i + self.input_steps - 1]
            
            if current_vol < vol_limit:
                if pred_change < -self.threshold:
                    pos = -1 # Short-Only
            
            # Update capital
            actual_ret = (test_prices[i + self.input_steps] - test_prices[i + self.input_steps - 1]) / test_prices[i + self.input_steps - 1]
            capital *= (1 + actual_ret * pos)
            equity.append(capital)
            
        return equity

if __name__ == "__main__":
    engine = ZenithEngine()
    zenith_equity = engine.run_backtest()
    
    # Load baseline for comparison
    baseline = pd.read_csv('results_1min_4mo.csv')['equity_short'].values
    
    plt.figure(figsize=(10, 6))
    plt.plot(baseline, label='Short-Only Baseline', color='red', alpha=0.5)
    plt.plot(zenith_equity, label='Project Zenith (GARCH Gated)', color='navy', linewidth=2)
    plt.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.title('Project Zenith: Production Performance Audit')
    plt.ylabel('Value ($)')
    plt.xlabel('Minutes')
    plt.legend()
    plt.grid(True)
    plt.savefig('zenith_performance.png')
    
    print(f"Zenith Final Value: ${zenith_equity[-1]:.2f}")
    print(f"Baseline Final Value: ${baseline[-1]:.2f}")
