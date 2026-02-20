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

def estimate_garch(returns, omega=1e-6, alpha=0.1, beta=0.8):
    n = len(returns)
    sigmas = np.zeros(n)
    sigmas[0] = np.max([np.var(returns), 1e-6])
    for t in range(1, n):
        sigmas[t] = omega + alpha * (returns[t-1]**2) + beta * sigmas[t-1]
    return np.sqrt(sigmas)

def process_stock(ticker, train_time_limit=600):
    csv_path = f'data/audit/stocks/{ticker}.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None
        
    print(f"\n>>> Deep Training Audit: {ticker} (Stocks) <<<")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split: Historical (< Feb 18) vs Live (>= Feb 18)
    split_date = pd.to_datetime('2026-02-18 00:00:00', utc=True)
    hist_df = df[df['timestamp'] < split_date].copy()
    live_df = df[df['timestamp'] >= split_date].copy()
    
    train_size = int(len(hist_df) * 0.8)
    train_df = hist_df.iloc[:train_size]
    val_df = hist_df.iloc[train_size:]
    
    scaler = StandardScaler()
    p_train_scaled = scaler.fit_transform(train_df['close'].values.reshape(-1, 1)).flatten()
    p_val_scaled = scaler.transform(val_df['close'].values.reshape(-1, 1)).flatten()
    p_live_scaled = scaler.transform(live_df['close'].values.reshape(-1, 1)).flatten()
    
    def create_sequences(data, seq_len=16, pred_len=4):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i : i + seq_len])
            y.append(data[i + seq_len : i + seq_len + pred_len])
        return np.array(X), np.array(y)
        
    X_train, y_train = create_sequences(p_train_scaled)
    X_val, y_val = create_sequences(p_val_scaled)
    
    model = FFNN()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=32)
    
    best_val_loss = float('inf')
    start_time = time.time()
    epoch = 0
    while time.time() - start_time < train_time_limit:
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if time.time() - start_time > train_time_limit: break
            
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                loss = criterion(model(bx), by)
                epoch_val_loss += loss.item()
        
        avg_val = epoch_val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f'zenith_stock_{ticker}_best.pth')
            
        if epoch % 10 == 0:
            print(f"  {ticker} | Epoch {epoch:03d} | Val Loss: {avg_val:.6f} | Time: {int(time.time()-start_time)}s")
        epoch += 1

    # Evaluation
    model.load_state_dict(torch.load(f'zenith_stock_{ticker}_best.pth'))
    model.eval()
    
    # Calculate GARCH on historical
    hist_prices_all = hist_df['close'].values
    hist_rets = np.diff(hist_prices_all) / hist_prices_all[:-1]
    hist_vols = estimate_garch(hist_rets)
    vol_limit = 1.5 * np.mean(hist_vols)
    
    live_prices_all = live_df['close'].values
    if len(live_prices_all) < 2:
        return {'ticker': ticker, 'roi': 0.0, 'trades': 0, 'val_loss': best_val_loss}
        
    full_prices = np.concatenate([hist_prices_all, live_prices_all])
    full_rets = np.diff(full_prices) / full_prices[:-1]
    full_vols = estimate_garch(full_rets)
    live_start_idx = len(hist_prices_all)
    
    capital = 100.0
    input_steps = 16
    trades_count = 0
    p_combined_scaled = np.concatenate([p_train_scaled[-input_steps:], p_live_scaled])
    
    for i in range(len(live_prices_all) - 1):
        x_window = p_combined_scaled[i : i + input_steps]
        x_t = torch.FloatTensor(x_window).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_t).numpy().flatten()
        
        pred_change = np.mean(pred) - x_window[-1]
        current_vol = full_vols[live_start_idx + i - 1]
        
        pos = 0
        if current_vol < vol_limit and pred_change < -0.0001:
            pos = -1
            trades_count += 1
            
        actual_ret = (live_prices_all[i+1] - live_prices_all[i]) / live_prices_all[i]
        capital *= (1 + actual_ret * pos)
        
    roi = capital - 100.0
    print(f"  {ticker} Result: ROI {roi:.2f}% | Trades: {trades_count}")
    return {'ticker': ticker, 'roi': roi, 'trades': trades_count, 'val_loss': best_val_loss}

def run_multi_stock_audit():
    tickers = ['NVDA', 'TSLA', 'AAPL', 'MSFT']
    results = []
    for ticker in tickers:
        res = process_stock(ticker, train_time_limit=600)
        if res:
            results.append(res)
            
    res_df = pd.DataFrame(results)
    res_df.to_csv('zenith_stock_audit_results.csv', index=False)
    print("\nMulti-Stock Audit Complete.")

if __name__ == "__main__":
    run_multi_stock_audit()
