import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Synthetic Data Generators
def gen_gaussian_walk(n=5000, mu=0.0001, sigma=0.01):
    rets = np.random.normal(mu, sigma, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return prices

def gen_martingale(n=5000, sigma=0.01):
    # Pure random walk (Arithmetic)
    steps = np.random.normal(0, sigma, n)
    prices = 100 + np.cumsum(steps)
    return prices

def gen_signal_plus_noise(n=5000, signal_freq=0.05, noise_std=0.5):
    t = np.arange(n)
    signal = 10 * np.sin(signal_freq * t)
    noise = np.random.normal(0, noise_std, n)
    prices = 100 + signal + noise
    return prices

def gen_jump_diffusion(n=5000, mu=0.0001, sigma=0.01, lambda_j=0.01, mu_j=-0.05, sigma_j=0.02):
    rets = np.random.normal(mu, sigma, n)
    jumps = np.random.poisson(lambda_j, n) * np.random.normal(mu_j, sigma_j, n)
    prices = 100 * np.exp(np.cumsum(rets + jumps))
    return prices

# 2. Zenith Implementation (Simplified for Synthetic)
def estimate_vol(rets, window=20):
    return pd.Series(rets).rolling(window).std().fillna(method='bfill').values

def run_zenith_synthetic(prices, name):
    rets = np.diff(prices) / prices[:-1]
    vol = estimate_vol(rets)
    vol_limit = 1.2 * np.mean(vol)
    
    # Mock model: Assume it "predicts" based on a lookback trend
    # In reality, this would be the FFNN. For audit, we test the GATE.
    # Signal: if 5-period momentum is negative, go short.
    momentum = pd.Series(prices).diff(5).shift(-5).fillna(0).values[:-1]
    
    capital_baseline = 100.0
    capital_zenith = 100.0
    equity_baseline = [100.0]
    equity_zenith = [100.0]
    
    for i in range(len(rets)):
        # Short-only signal
        pos_signal = -1 if momentum[i] < 0 else 0
        
        # Gating
        gated = vol[i] > vol_limit
        pos_zenith = pos_signal if not gated else 0
        
        # Backtest rets
        capital_baseline *= (1 + rets[i] * pos_signal)
        capital_zenith *= (1 + rets[i] * pos_zenith)
        
        equity_baseline.append(capital_baseline)
        equity_zenith.append(capital_zenith)
        
    return equity_baseline, equity_zenith

# 3. Execution Loop
if __name__ == "__main__":
    np.random.seed(42)
    models = {
        "Gaussian Walk (GBM)": gen_gaussian_walk(),
        "Martingale": gen_martingale(),
        "Signal + Noise": gen_signal_plus_noise(),
        "Jump-Diffusion": gen_jump_diffusion()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    results = []
    
    for i, (name, prices) in enumerate(models.items()):
        eb, ez = run_zenith_synthetic(prices, name)
        axes[i].plot(eb, label='Un-gated Short', alpha=0.6, color='red')
        axes[i].plot(ez, label='Zenith (Gated)', linewidth=2, color='blue')
        axes[i].set_title(name)
        axes[i].legend()
        results.append({"Model": name, "Baseline": eb[-1], "Zenith": ez[-1]})
        
    plt.tight_layout()
    plt.savefig('zenith_synthetic_audit.png')
    
    df_res = pd.DataFrame(results)
    df_res.to_csv('zenith_synthetic_results.csv', index=False)
    print(df_res)
