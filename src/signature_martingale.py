import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import sys

def get_signature_2d_depth2(path):
    """
    Manual signature computation for 2D path (t, x) depth 2.
    Path is (N, 2).
    Degree 1: integrals of dX_i (increment)
    Degree 2: integrals of X_i dX_j
    """
    N = path.shape[0]
    # Increments
    inc = np.diff(path, axis=0)
    
    # Degree 1 (2 components)
    d1 = path[-1] - path[0]
    
    # Degree 2 (4 components: 11, 12, 21, 22)
    # Integral of X_i dX_j ~ sum_{k} X_i(k) * (X_j(k+1) - X_j(k))
    # Using midpoint rule for better accuracy
    mid_path = (path[:-1] + path[1:]) / 2
    
    d2 = np.zeros(4)
    idx = 0
    for i in range(2):
        for j in range(2):
            d2[idx] = np.sum(mid_path[:, i] * inc[:, j])
            idx += 1
            
    return np.concatenate([d1, d2])

def prepare_signature_dataset(csv_path, window=20):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # Cast Close to numeric if needed
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    
    rets = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
    cum_rets = np.cumsum(rets)
    
    X, y = [], []
    for i in range(len(cum_rets) - window - 1):
        # Path: (time, cum_ret)
        # Time is normalized to [0, 1] for the window
        times = np.linspace(0, 1, window + 1)
        prices = cum_rets[i : i + window + 1]
        prices = prices - prices[0] # Start at 0
        
        path = np.column_stack([times, prices])
        sig = get_signature_2d_depth2(path)
        X.append(sig)
        y.append(rets[i + window]) # Predict next return
        
    return np.array(X), np.array(y)

class SigNet(nn.Module):
    def __init__(self, input_dim=6):
        super(SigNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_experiment(symbol):
    csv_path = f'data/signature_experiment/{symbol}_4y.csv'
    X, y = prepare_signature_dataset(csv_path)
    
    # Train/Test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    y_train, y_test = torch.FloatTensor(y[:split]).reshape(-1, 1), torch.FloatTensor(y[split:]).reshape(-1, 1)
    
    model = SigNet()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Training {symbol} for 120s...")
    start_time = time.time()
    while time.time() - start_time < 120:
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy().flatten()
        test_actual = y_test.numpy().flatten()
        
    # Simulation
    capital = 100.0
    for i in range(len(test_preds)):
        p = test_preds[i]
        act = test_actual[i]
        if p > 0.0001: capital += capital * act - 0.01
        elif p < -0.0001: capital -= capital * act + 0.01
        if capital <= 0: capital = 0; break
        
    print(f"  {symbol} Final Capital: ${capital:.2f}")
    return capital

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
    results = {}
    for s in symbols:
        results[s] = run_experiment(s)
    
    with open('signature_results.txt', 'w') as f:
        for s, c in results.items():
            f.write(f"{s} {c:.2f}\n")
