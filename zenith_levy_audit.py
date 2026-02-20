import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import levy_stable

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

def estimate_levy_alpha(returns):
    """Fast quantile-based estimation of the stability index alpha."""
    try:
        q = np.percentile(returns, [5, 25, 75, 95])
        # v: spread ratio (95-5) / (75-25)
        # For Gaussian: v ~ 2.44 / 1.35 ~ 1.8
        # For Cauchy (alpha=1): v ~ 12.7 / 2.0 ~ 6.35
        v = (q[3] - q[0]) / (q[2] - q[1])
        
        # Rough linear interpolation for alpha in [1, 2]
        # v=1.8 -> alpha=2.0
        # v=6.3 -> alpha=1.0
        if v <= 1.8: return 2.0
        alpha = 2.0 - (v - 1.8) * (1.0 / (6.3 - 1.8))
        return max(1.0, alpha)
    except:
        return 2.0

def process_asset(ticker, train_time_limit=120):
    csv_path = f'data/audit/{ticker}/1m.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None
        
    print(f"\n>>> Lévy Stable Audit: {ticker} <<<")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split: Historical (< Feb 18) vs Live (>= Feb 18)
    split_date = pd.to_datetime('2026-02-18 00:00:00', utc=False)
    # Ensure UTC-awareness if data is UTC
    if df['timestamp'].dt.tz is None:
        split_date = pd.to_datetime('2026-02-18 00:00:00')
    
    hist_df = df[df['timestamp'] < split_date].copy()
    live_df = df[df['timestamp'] >= split_date].copy()
    
    if len(live_df) < 5:
        print(f"Warning: {ticker} has insufficient live data.")
    
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
            torch.save(model.state_dict(), f'zenith_levy_{ticker}_best.pth')
        
        if epoch % 5 == 0:
            print(f"  {ticker} | Epoch {epoch:03d} | Val Loss: {avg_val:.6f} | Time: {int(time.time()-start_time)}s")
        epoch += 1

    # Evaluation
    model.load_state_dict(torch.load(f'zenith_levy_{ticker}_best.pth'))
    model.eval()
    
    # Estimate Lévy Alpha on historical returns
    hist_prices_all = hist_df['close'].values
    hist_rets = np.diff(hist_prices_all) / hist_prices_all[:-1]
    
    print(f"  Estimating Lévy Alpha for {ticker}...")
    # Use a small subsample for fitting speed
    # levy_stable.fit is O(N) but the constant is very large for MLE.
    # 1000 points is enough for a regional estimate.
    fit_sample = np.random.choice(hist_rets, size=min(1000, len(hist_rets)), replace=False)
    alpha = estimate_levy_alpha(fit_sample)
    print(f"  Alpha: {alpha:.4f} (Gaussian=2.0)")
    
    # Risk decision based on Alpha-stability
    # If alpha is low, we need higher conviction for trades.
    conviction_threshold = 0.001 / (alpha / 2.0) # Adaptive threshold
    
    live_prices_all = live_df['close'].values
    capital = 100.0
    input_steps = 16
    trades_count = 0  # Now actual round-trip trades
    exposure_mins = 0
    p_combined_scaled = np.concatenate([p_train_scaled[-input_steps:], p_live_scaled])
    
    current_pos = 0 # 0 or -1
    fee_rate = 0.0005 # 5 bps round-trip
    
    # Track trades for logging
    trade_logs = []
    
    for i in range(len(live_prices_all) - 1):
        x_window = p_combined_scaled[i : i + input_steps]
        x_t = torch.FloatTensor(x_window).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_t).numpy().flatten()
        
        pred_change = np.mean(pred) - x_window[-1]
        
        # Position: Short-Only
        target_pos = 0
        if pred_change < -conviction_threshold:
            target_pos = -1
            
        # Check for Entry/Exit to apply fees
        if target_pos == -1 and current_pos == 0:
            # Entry
            trades_count += 1
            capital *= (1 - fee_rate) # Deduct fee on entry (simplification for round-trip)
        elif target_pos == 0 and current_pos == -1:
            # Exit
            pass # Fee already deducted at entry (round-trip)
            
        current_pos = target_pos
        if current_pos == -1:
            exposure_mins += 1
            
        actual_ret = (live_prices_all[i+1] - live_prices_all[i]) / live_prices_all[i]
        old_cap = capital
        capital *= (1 + actual_ret * current_pos)
        
        if current_pos == -1:
            trade_logs.append({
                'timestamp': live_df.iloc[i]['timestamp'],
                'price': live_prices_all[i],
                'pred_change': pred_change,
                'roi_step': (capital / old_cap - 1) * 100
            })
            
    roi = (capital / 100.0 - 1) * 100
    print(f"  {ticker} Result: ROI {roi:.4f}% | Trades: {trades_count} | Exposure: {exposure_mins}m | Alpha: {alpha:.4f}")
    
    # Save trade logs
    pd.DataFrame(trade_logs).to_csv(f'zenith_levy_trades_{ticker}.csv', index=False)
    
    return {
        'ticker': ticker, 
        'roi': roi, 
        'trades': trades_count, 
        'exposure_mins': exposure_mins,
        'val_loss': best_val_loss,
        'alpha': alpha
    }

def run_levy_audit():
    tickers = ['DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT']
    results = []
    for ticker in tickers:
        res = process_asset(ticker, train_time_limit=120)
        if res:
            results.append(res)
            
    res_df = pd.DataFrame(results)
    res_df.to_csv('zenith_levy_audit_results.csv', index=False)
    print("\nLévy Stable Audit Complete.")

if __name__ == "__main__":
    run_levy_audit()
