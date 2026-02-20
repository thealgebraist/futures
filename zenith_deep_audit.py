import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
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
    sigmas[0] = np.max([np.var(returns), 1e-6])
    for t in range(1, n):
        sigmas[t] = omega + alpha * (returns[t-1]**2) + beta * sigmas[t-1]
    return np.sqrt(sigmas)

def run_deep_audit():
    coin = 'SOLUSDT'
    csv_path = f'data/audit/{coin}/1m.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"\n>>> Deep Training Audit: {coin} <<<")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split: Historical (< Feb 18) vs Live (>= Feb 18)
    split_date = pd.to_datetime('2026-02-18 00:00:00')
    hist_df = df[df['timestamp'] < split_date].copy()
    live_df = df[df['timestamp'] >= split_date].copy()
    
    # Further split Historical into 80% Train, 20% Validation
    train_size = int(len(hist_df) * 0.8)
    train_df = hist_df.iloc[:train_size]
    val_df = hist_df.iloc[train_size:]
    
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Live:  {len(live_df)} rows")

    # 1. Scaling
    scaler = StandardScaler()
    train_prices = train_df['close'].values.reshape(-1, 1)
    val_prices = val_df['close'].values.reshape(-1, 1)
    live_prices = live_df['close'].values.reshape(-1, 1)
    
    p_train_scaled = scaler.fit_transform(train_prices).flatten()
    p_val_scaled = scaler.transform(val_prices).flatten()
    p_live_scaled = scaler.transform(live_prices).flatten()
    
    def create_sequences(data, seq_len=16, pred_len=4):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i : i + seq_len])
            y.append(data[i + seq_len : i + seq_len + pred_len])
        return np.array(X), np.array(y)
        
    X_train, y_train = create_sequences(p_train_scaled)
    X_val, y_val = create_sequences(p_val_scaled)
    
    # 2. Training (1200s)
    model = FFNN()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    print(f"  Starting 20-minute training loop...")
    start_time = time.time()
    epoch = 0
    while time.time() - start_time < 1200:
        model.train()
        epoch_train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            if time.time() - start_time > 1200: break
            
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                loss = criterion(model(bx), by)
                epoch_val_loss += loss.item()
        
        avg_train = epoch_train_loss / len(train_loader)
        avg_val = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'zenith_deep_best.pth')
            
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:03d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | Time: {int(time.time()-start_time)}s")
        epoch += 1

    # Load Best Model
    model.load_state_dict(torch.load('zenith_deep_best.pth'))
    model.eval()
    
    # 3. Live Simulation
    # Calculate GARCH on historical mean to set threshold
    hist_prices_all = hist_df['close'].values
    hist_rets = np.diff(hist_prices_all) / hist_prices_all[:-1]
    hist_vols = estimate_garch(hist_rets)
    vol_limit = 1.5 * np.mean(hist_vols)
    
    live_prices_all = live_df['close'].values
    full_prices = np.concatenate([hist_prices_all, live_prices_all])
    full_rets = np.diff(full_prices) / full_prices[:-1]
    full_vols = estimate_garch(full_rets)
    
    # Test offset in full_vols
    live_start_idx = len(hist_prices_all)
    
    input_steps = 16
    capital = 100.0
    equity = [capital]
    trades = []
    
    # Prepare combined scaled data for sequencing into live
    p_combined_scaled = np.concatenate([p_train_scaled[-input_steps:], p_live_scaled])
    
    for i in range(len(live_prices_all) - 1):
        x_window = p_combined_scaled[i : i + input_steps]
        x_t = torch.FloatTensor(x_window).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x_t).numpy().flatten()
            
        last_val_scaled = x_window[-1]
        pred_change = np.mean(pred) - last_val_scaled
        
        # GARCH Gating
        current_vol = full_vols[live_start_idx + i - 1]
        pos = 0
        gated = False
        if current_vol < vol_limit:
            if pred_change < -0.0001:
                pos = -1
        else:
            gated = True
            
        actual_ret = (live_prices_all[i+1] - live_prices_all[i]) / live_prices_all[i]
        capital *= (1 + actual_ret * pos)
        equity.append(capital)
        
        if pos != 0:
            trades.append({
                'timestamp': live_df.iloc[i]['timestamp'],
                'price_usd': live_prices_all[i],
                'pred_change': pred_change,
                'vol': current_vol,
                'gated': gated,
                'roi_step': actual_ret * pos
            })
            
    # Export
    pd.DataFrame(trades).to_csv('zenith_deep_trades.csv', index=False)
    
    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Zenith Deep Training: Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.yscale('log')
    plt.savefig('zenith_deep_losses.png')
    
    print(f"\n--- DEEP AUDIT COMPLETE ---")
    print(f"  Final Live Equity: ${capital:.2f}")
    print(f"  Trades Executed: {len(trades)}")
    
    # Save statistics for LaTeX
    with open('zenith_deep_stats.txt', 'w') as f:
        f.write(f"training_time: 1200\n")
        f.write(f"final_equity: {capital:.2f}\n")
        f.write(f"num_trades: {len(trades)}\n")
        f.write(f"best_val_loss: {best_val_loss:.8f}\n")

if __name__ == "__main__":
    run_deep_audit()
