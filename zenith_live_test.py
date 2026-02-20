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

# --- Live Audit Function ---
def process_live_audit(coin, interval='1m'):
    csv_path = f'data/audit/{coin}/{interval}.csv'
    
    # Wait loop for file existence (increased to sync with downloader)
    wait_attempts = 0
    while not os.path.exists(csv_path) and wait_attempts < 720: # 1 hour wait
        if wait_attempts % 60 == 0:
            print(f"Waiting for {csv_path} (Attempt {wait_attempts})...")
        time.sleep(5)
        wait_attempts += 1
        
    if not os.path.exists(csv_path):
        print(f"Skipping {coin}: Data file not found.")
        return None
        
    print(f"\n>>> Live Audit: {coin} <<<")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split: Before today vs Today
    split_date = pd.to_datetime('2026-02-18 00:00:00')
    train_df = df[df['timestamp'] < split_date]
    test_df = df[df['timestamp'] >= split_date]
    
    if len(train_df) < 5000:
        print(f"  Insufficient training data ({len(train_df)} rows).")
        return None
    if len(test_df) < 10:
        print(f"  Insufficient live data ({len(test_df)} rows).")
        return None
        
    print(f"  Training on {len(train_df)} rows (before 2026-02-18)")
    print(f"  Simulating on {len(test_df)} rows (from 0:00 today)")

    # 1. Prepare Training Data (80% of historical for validation, but we'll use all for 120s train)
    train_prices = train_df['close'].values
    test_prices = test_df['close'].values
    
    scaler = StandardScaler()
    p_train_scaled = scaler.fit_transform(train_prices.reshape(-1, 1)).flatten()
    p_test_scaled = scaler.transform(test_prices.reshape(-1, 1)).flatten()
    
    input_steps = 16
    output_steps = 4
    X_train, y_train = [], []
    for i in range(len(p_train_scaled) - input_steps - output_steps + 1):
        X_train.append(p_train_scaled[i : i + input_steps])
        y_train.append(p_train_scaled[i + input_steps : i + input_steps + output_steps])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 2. Train (120s)
    model = FFNN()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_t = torch.FloatTensor(X_train)
    train_y_t = torch.FloatTensor(y_train)
    loader = DataLoader(TensorDataset(train_t, train_y_t), batch_size=32, shuffle=True)
    
    print(f"  Training for 120s...")
    start_train = time.time()
    while time.time() - start_train < 120:
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    # 3. Simulate Live Performance
    model.eval()
    
    # Aligned vols (GARCH estimation on full historical + live)
    full_prices = np.concatenate([train_prices, test_prices])
    rets = np.diff(full_prices) / full_prices[:-1]
    vols = estimate_garch(rets)
    vol_limit = 1.5 * np.mean(vols[:len(train_prices)])
    
    # Test indices: correspond to test_prices
    test_start_idx = len(train_prices)
    
    capital = 100.0
    equity = [capital]
    trades = []
    
    # Need windows for test set
    p_combined_scaled = np.concatenate([p_train_scaled[-input_steps:], p_test_scaled])
    
    for i in range(len(test_prices) - 1):
        x_window = p_combined_scaled[i : i + input_steps]
        x_t = torch.FloatTensor(x_window).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x_t).numpy().flatten()
            
        last_val_scaled = x_window[-1]
        pred_change = np.mean(pred) - last_val_scaled
        
        # GARCH Gating
        current_vol = vols[test_start_idx + i - 1]
        pos = 0
        gated = False
        if current_vol < vol_limit:
            if pred_change < -0.0001:
                pos = -1 # Zenith Short-Only
        else:
            gated = True
        
        if pos != 0:
            trades.append({
                'timestamp': test_df.iloc[i]['timestamp'],
                'price': test_prices[i],
                'pred_change': pred_change,
                'vol': current_vol,
                'gated': gated,
                'position': 'SHORT'
            })
        
        actual_ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]
        capital *= (1 + actual_ret * pos)
        equity.append(capital)
        
    print(f"  Live PnL: ${capital:.2f}")
    
    # Export trades for this coin
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f'{coin.lower()}_trades.csv', index=False)
        print(f"  Exported {len(trades)} trades to {coin.lower()}_trades.csv")

    return {
        'coin': coin, 
        'final_equity': capital,
        'roi': capital - 100.0,
        'steps': len(test_prices)
    }

if __name__ == "__main__":
    top_8 = ['SOLUSDT']
    
    # Load existing results
    report_file = 'zenith_live_audit_report.csv'
    if os.path.exists(report_file):
        existing_results = pd.read_csv(report_file)
        completed_coins = existing_results['coin'].tolist()
    else:
        existing_results = pd.DataFrame()
        completed_coins = []
        
    results = existing_results.to_dict('records')
    
    for coin in top_8:
        if coin in completed_coins:
            print(f"Skipping {coin}: Already completed.")
            continue
            
        res = process_live_audit(coin)
        if res:
            results.append(res)
            # Full rewrite for intermediate safety
            pd.DataFrame(results).to_csv(report_file, index=False)
            
    print("\n--- LIVE AUDIT COMPLETE ---")
    summary = pd.DataFrame(results)
    print(summary)
