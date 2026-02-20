import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --- Bayesian Layer Implementation ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Mean and variance parameters (rho is for sigma)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.1, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).fill_(-3.0))
        
    def forward(self, x):
        # Sample weights: w = mu + sigma * epsilon
        # sigma = softplus(rho)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_epsilon = torch.randn_like(weight_sigma)
        weight = self.weight_mu + weight_sigma * weight_epsilon
        
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_epsilon = torch.randn_like(bias_sigma)
        bias = self.bias_mu + bias_sigma * bias_epsilon
        
        return F.linear(x, weight, bias)

class BayesianFFNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=512, output_dim=4):
        super(BayesianFFNN, self).__init__()
        self.l1 = BayesianLinear(input_dim, hidden_dim)
        self.l2 = BayesianLinear(hidden_dim, hidden_dim)
        self.l3 = BayesianLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

def process_asset_bayesian(ticker, train_time_limit=120):
    csv_path = f'data/audit/{ticker}/1m.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None
        
    print(f"\n>>> Bayesian Weight Audit: {ticker} <<<")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split: Historical (< Feb 18) vs Live (>= Feb 18)
    split_date = pd.to_datetime('2026-02-18 00:00:00')
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
    
    model = BayesianFFNN()
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
            # In Bayesian, each forward is a different sample of weights
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if time.time() - start_time > train_time_limit: break
            
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                # During evaluation, we still sample weights per the user's request
                output = model(bx)
                loss = criterion(output, by)
                epoch_val_loss += loss.item()
        
        avg_val = epoch_val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f'zenith_bayesian_{ticker}_best.pth')
            
        if epoch % 5 == 0:
            print(f"  {ticker} | Epoch {epoch:03d} | Val Loss: {avg_val:.6f} | Time: {int(time.time()-start_time)}s")
        epoch += 1

    # Evaluation
    model.load_state_dict(torch.load(f'zenith_bayesian_{ticker}_best.pth'))
    model.eval()
    
    live_prices_all = live_df['close'].values
    capital = 100.0
    input_steps = 16
    trades_count = 0
    p_combined_scaled = np.concatenate([p_train_scaled[-input_steps:], p_live_scaled])
    
    trade_logs = []
    
    for i in range(len(live_prices_all) - 1):
        x_window = p_combined_scaled[i : i + input_steps]
        x_t = torch.FloatTensor(x_window).unsqueeze(0)
        
        # Stochastic Inference: User asked for weights to be random each time evaluated.
        # We can take multiple samples to see uncertainty
        samples = 10
        preds = []
        with torch.no_grad():
            for _ in range(samples):
                preds.append(model(x_t).numpy().flatten())
        
        preds = np.array(preds)
        mean_pred = np.mean(preds, axis=0)
        uncertainty = np.std(preds)
        
        pred_change = np.mean(mean_pred) - x_window[-1]
        
        # Decision: Short-Only
        # We can use uncertainty to gate trades: high uncertainty = No Trade.
        pos = 0
        if pred_change < -0.0005 and uncertainty < 0.05:
            pos = -1
            trades_count += 1
            
        actual_ret = (live_prices_all[i+1] - live_prices_all[i]) / live_prices_all[i]
        old_cap = capital
        capital *= (1 + actual_ret * pos)
        
        if pos == -1:
            trade_logs.append({
                'timestamp': live_df.iloc[i]['timestamp'],
                'price': live_prices_all[i],
                'pred_change': pred_change,
                'uncertainty': uncertainty,
                'roi_step': (capital / old_cap - 1) * 100
            })
            
    roi = (capital / 100.0 - 1) * 100
    print(f"  {ticker} Result: ROI {roi:.4f}% | Trades: {trades_count}")
    
    pd.DataFrame(trade_logs).to_csv(f'zenith_bayesian_trades_{ticker}.csv', index=False)
    
    return {
        'ticker': ticker, 
        'roi': roi, 
        'trades': trades_count, 
        'val_loss': best_val_loss
    }

def run_bayesian_audit():
    tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    results = []
    for ticker in tickers:
        res = process_asset_bayesian(ticker, train_time_limit=120)
        if res:
            results.append(res)
            
    res_df = pd.DataFrame(results)
    res_df.to_csv('zenith_bayesian_audit_results.csv', index=False)
    print("\nBayesian Weight Audit Complete.")

if __name__ == "__main__":
    run_bayesian_audit()
