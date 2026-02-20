import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --- Model Definition ---
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

# --- GARCH Filter ---
def estimate_garch(returns, omega=1e-6, alpha=0.1, beta=0.8):
    n = len(returns)
    sigmas = np.zeros(n)
    sigmas[0] = np.var(returns)
    for t in range(1, n):
        sigmas[t] = omega + alpha * (returns[t-1]**2) + beta * sigmas[t-1]
    return np.sqrt(sigmas)

# --- Pipeline Function ---
def process_configuration(coin, interval):
    csv_path = f'data/audit/{coin}/{interval}.csv'
    if not os.path.exists(csv_path):
        print(f"Skipping {coin} {interval}: Data not found.")
        return None
        
    print(f"\n>>> Processing {coin} | {interval} <<<")
    df = pd.read_csv(csv_path)
    prices = df['close'].values
    prices = prices[~np.isnan(prices)]
    
    if len(prices) < 1000:
        print(f"Skipping {coin} {interval}: Insufficient data ({len(prices)} rows).")
        return None

    # 1. Prepare Data
    scaler = StandardScaler()
    p_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    input_steps = 16
    output_steps = 4
    X, y = [], []
    for i in range(len(p_scaled) - input_steps - output_steps + 1):
        X.append(p_scaled[i : i + input_steps])
        y.append(p_scaled[i + input_steps : i + input_steps + output_steps])
    X, y = np.array(X), np.array(y)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 2. Train (120s)
    model = FFNN()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_t = torch.FloatTensor(X_train)
    train_y_t = torch.FloatTensor(y_train)
    loader = DataLoader(TensorDataset(train_t, train_y_t), batch_size=32, shuffle=True)
    
    print(f"  Training for 120s...")
    start_train = time.time()
    epochs = 0
    while time.time() - start_train < 120:
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        epochs += 1
    
    # 3. Backtest (Project Zenith Logic)
    model.eval()
    test_prices = prices[split_idx + input_steps:] # Matching the test window
    test_p_scaled = p_scaled[split_idx + input_steps:]
    
    # GARCH logic on full series to get historical mean
    rets = np.diff(prices) / prices[:-1]
    vols = estimate_garch(rets)
    vol_limit = 1.5 * np.mean(vols)
    
    # Backtest Window
    test_vols = vols[split_idx + input_steps - 1 : -1] # aligned with test prices
    
    capital = 100.0
    equity = [capital]
    
    # Vectorized batch prediction for speed
    X_test_batch = torch.FloatTensor(X_test)
    with torch.no_grad():
        preds = model(X_test_batch).numpy()
    
    # Simulation
    for i in range(len(preds) - 1):
        last_val_scaled = X_test[i][-1]
        pred_change = np.mean(preds[i]) - last_val_scaled
        
        pos = 0
        if test_vols[i] < vol_limit:
            if pred_change < -0.0001:
                pos = -1
        
        actual_ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]
        capital *= (1 + actual_ret * pos)
        equity.append(capital)
        
    final_val = equity[-1]
    roi = (final_val - 100.0)
    print(f"  Done. Final Value: ${final_val:.2f} | ROI: {roi:.2f}%")
    
    return {
        'coin': coin,
        'interval': interval,
        'final_value': final_val,
        'roi': roi,
        'mse': criterion(torch.FloatTensor(preds), torch.FloatTensor(y_test)).item()
    }

if __name__ == "__main__":
    coins = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 
        'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'SHIBUSDT', 'DOTUSDT', 
        'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'PEPEUSDT', 'WIFUSDT', 'RENDERUSDT'
    ]
    intervals = ['1m', '10m', '1h']
    
    all_results = []
    
    # Check if download is still running (very rough check)
    downloader_file = 'download_audit_data.py'
    
    for coin in coins:
        for interval in intervals:
            csv_path = f'data/audit/{coin}/{interval}.csv'
            
            # Wait loop for file existence
            wait_attempts = 0
            while not os.path.exists(csv_path) and wait_attempts < 60: # wait up to 10 mins
                print(f"Waiting for {csv_path} (Attempt {wait_attempts})...")
                time.sleep(10)
                wait_attempts += 1
                
            if os.path.exists(csv_path):
                res = process_configuration(coin, interval)
                if res:
                    all_results.append(res)
                    pd.DataFrame(all_results).to_csv('zenith_audit_results.csv', index=False)
            else:
                print(f"Timed out waiting for {csv_path}. Skipping.")
    
    print("\n--- GLOBAL AUDIT COMPLETE ---")
    summary = pd.DataFrame(all_results)
    print(summary)
