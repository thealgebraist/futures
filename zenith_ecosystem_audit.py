import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import concurrent.futures

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
    try:
        q = np.percentile(returns, [5, 25, 75, 95])
        v = (q[3] - q[0]) / (q[2] - q[1])
        if v <= 1.8: return 2.0
        alpha = 2.0 - (v - 1.8) * (1.0 / (6.3 - 1.8))
        return max(1.0, alpha)
    except:
        return 2.0

def process_asset_ecosystem(ticker, train_time_limit=60):
    csv_path = f'data/audit/{ticker}/10m.csv'
    if not os.path.exists(csv_path):
        return None
        
    print(f"Starting {ticker} (60s Audit)...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 6mo data, split 80/20 for Train/Val, Eval on last 1000 candles as "Live"
    live_start_idx = len(df) - 1000
    hist_df = df.iloc[:live_start_idx]
    live_df = df.iloc[live_start_idx:]
    
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
            # Save uniquely per process
            torch.save(model.state_dict(), f'data/audit/{ticker}/eco_best.last.pth')
        epoch += 1

    # Evaluation
    model.load_state_dict(torch.load(f'data/audit/{ticker}/eco_best.last.pth'))
    model.eval()
    
    hist_prices_all = hist_df['close'].values
    hist_rets = np.diff(hist_prices_all) / hist_prices_all[:-1]
    fit_sample = np.random.choice(hist_rets, size=min(1000, len(hist_rets)), replace=False)
    alpha = estimate_levy_alpha(fit_sample)
    
    conviction_threshold = 0.001 / (alpha / 2.0)
    live_prices_all = live_df['close'].values
    capital = 100.0
    input_steps = 16
    trades_count = 0
    exposure_mins = 0
    current_pos = 0
    fee_rate = 0.0005 # 5 bps round-trip
    
    p_combined_scaled = np.concatenate([p_train_scaled[-input_steps:], p_live_scaled])
    
    for i in range(len(live_prices_all) - 1):
        x_window = p_combined_scaled[i : i + input_steps]
        x_t = torch.FloatTensor(x_window).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_t).numpy().flatten()
        
        pred_change = np.mean(pred) - x_window[-1]
        target_pos = 0
        if pred_change < -conviction_threshold:
            target_pos = -1
            
        if target_pos == -1 and current_pos == 0:
            trades_count += 1
            capital *= (1 - fee_rate)
            
        current_pos = target_pos
        if current_pos == -1:
            exposure_mins += 10 # 10m intervals
            
        actual_ret = (live_prices_all[i+1] - live_prices_all[i]) / live_prices_all[i]
        capital *= (1 + actual_ret * current_pos)
            
    roi = (capital / 100.0 - 1) * 100
    print(f"Finished {ticker}: ROI {roi:.4f}%, Alpha {alpha:.4f}")
    return {'ticker': ticker, 'roi': roi, 'trades': trades_count, 'exposure': exposure_mins, 'alpha': alpha}

def run_ecosystem_audit():
    MID_32 = [
        'LINKUSDT', 'POLUSDT', 'UNIUSDT', 'NEARUSDT', 'LDOUSDT', 
        'ICPUSDT', 'FILUSDT', 'ARBUSDT', 'OPUSDT', 'TIAUSDT', 
        'RENDERUSDT', 'FETUSDT', 'GRTUSDT', 'AAVEUSDT', 'CRVUSDT', 
        'SNXUSDT', 'MKRUSDT', 'DYDXUSDT', 'IMXUSDT', 'STXUSDT', 
        'KASUSDT', 'INJUSDT', 'SEIUSDT', 'SUIUSDT', 'APTUSDT', 
        'ARUSDT', 'THETAUSDT', 'FLOWUSDT', 'EGLDUSDT', 'ALGOUSDT', 
        'HBARUSDT', 'VETUSDT'
    ]
    LOW_32 = [
        'ANKRUSDT', 'CHZUSDT', 'ENJUSDT', 'BATUSDT', 'KAVAUSDT', 
        'ZILUSDT', 'RVNUSDT', 'SCUSDT', 'HOTUSDT', 'ONEUSDT', 
        'IOTAUSDT', 'ONTUSDT', 'QTUMUSDT', 'IOSTUSDT', 'ZRXUSDT', 
        'OMGUSDT', 'GLMUSDT', 'SXPUSDT', 'ALPHAUSDT', 'AUDIOUSDT', 
        'BANDUSDT', 'COTIUSDT', 'DGBUSDT', 'FUNUSDT', 'KNCUSDT', 
        'LRCUSDT', 'MTLUSDT', 'NMRUSDT', 'OGNUSDT', 'RAYUSDT', 
        'REQUSDT', 'STORJUSDT'
    ]
    
    all_tickers = MID_32 + LOW_32
    results = []
    
    print(f">>> Project Zenith: 64-Asset Ecosystem Audit (6 Months @ 10m) <<<")
    # Parallelize with 8 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(process_asset_ecosystem, ticker): ticker for ticker in all_tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                results.append(res)
                
    pdf_results = pd.DataFrame(results)
    pdf_results.to_csv('zenith_ecosystem_results.csv', index=False)
    print("\nEcosystem Audit Complete. Results saved to zenith_ecosystem_results.csv")

if __name__ == "__main__":
    run_ecosystem_audit()
