import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
from torch.nn.utils import weight_norm

def estimate_levy_alpha(returns):
    try:
        returns = returns[np.isfinite(returns)]
        if len(returns) < 100: return 2.0
        q = np.percentile(returns, [5, 25, 75, 95])
        v = (q[3] - q[0]) / (q[2] - q[1])
        if v <= 1.8: return 2.0
        alpha = 2.0 - (v - 1.8) * (1.0 / (6.3 - 1.8))
        return max(1.0, min(2.0, alpha))
    except:
        return 2.0

def calculate_martingale_ratio(prices, lags=[2, 5, 10]):
    """Variance Ratio Test as Martingale Indicator"""
    try:
        returns = np.diff(np.log(prices))
        vr_results = []
        for k in lags:
            rolling_sum = pd.Series(returns).rolling(window=k).sum().dropna()
            var_k = rolling_sum.var()
            var_1 = returns.var()
            vr = var_k / (k * var_1 + 1e-8)
            vr_results.append(vr)
        return np.mean(vr_results) # 1.0 = Random Walk
    except:
        return 1.0

class ZenithFFNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=1):
        super(ZenithFFNN, self).__init__()
        self.width = hidden_dim
        self.fc1 = weight_norm(nn.Linear(input_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.fc1.weight, a=np.sqrt(5))
        with torch.no_grad():
            self.fc1.weight.data *= (1.0 / np.sqrt(self.width))
        self.relu = nn.ReLU()
        self.fc2 = weight_norm(nn.Linear(hidden_dim, output_dim))
        nn.init.kaiming_uniform_(self.fc2.weight, a=np.sqrt(5))
        with torch.no_grad():
            self.fc2.weight.data *= (1.0 / self.width)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_gns(model, criterion, data, target):
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    g = torch.cat(grads)
    g_sq = torch.norm(g)**2
    v = torch.var(g)
    return v / (g_sq + 1e-8)

def train_asset_comp(ticker):
    print(f"Audit Start: {ticker}...", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    path = f"/Users/anders/projects/futures/data/audit/{ticker}/10m.csv"
    if not os.path.exists(path):
        print(f"Data missing for {ticker}")
        return None
    
    df = pd.read_csv(path)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['high'] - df['low']
    df = df.dropna()
    
    # Statistical Metrics
    alpha = estimate_levy_alpha(df['returns'].values)
    mr_ratio = calculate_martingale_ratio(df['close'].values)
    
    # Deep Learning Prep
    features = torch.tensor(df[['open', 'high', 'low', 'volume', 'returns', 'volatility']].values, dtype=torch.float32).to(device)
    targets = torch.tensor(df['close'].shift(-1).dropna().values, dtype=torch.float32).view(-1, 1).to(device)
    features = features[:len(targets)]
    
    mean, std = features.mean(dim=0), features.std(dim=0) + 1e-8
    features = (features - mean) / std
    
    model = ZenithFFNN(hidden_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RAdam(model.parameters(), lr=0.001)
    
    # GNS Batch Derivation
    gns = get_gns(model, criterion, features[:32], targets[:32])
    batch_size = int(max(32, min(512, 32 * (1 + gns.item()))))
    
    # Core Train (60s)
    start_time = time.time()
    while time.time() - start_time < 60:
        indices = torch.randint(0, len(features), (batch_size,))
        optimizer.zero_grad()
        loss = criterion(model(features[indices]), targets[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    final_loss = criterion(model(features), targets).item()
    print(f"Audit End: {ticker} | Loss: {final_loss:.6f} | Alpha: {alpha:.2f} | MR: {mr_ratio:.2f}")
    
    return {
        'ticker': ticker,
        'final_loss': final_loss,
        'alpha': alpha,
        'martingale_ratio': mr_ratio,
        'batch_size': batch_size
    }

if __name__ == "__main__":
    assets = [
        '1000PEPEUSDT', '1000BONKUSDT', '1000SHIBUSDT', 'GRTUSDT', 'ALGOUSDT', 'MANAUSDT', 'HBARUSDT', 'DOGEUSDT', 
        'POLUSDT', 'APEUSDT', 'XLMUSDT', 'SUSHIUSDT', 'ONDOUSDT', 'ADAUSDT', 'TRXUSDT', 'WLDUSDT', 
        'XTZUSDT', 'SUIUSDT', 'FILUSDT', 'DOTUSDT', 'NEARUSDT', 'TONUSDT', 'XRPUSDT', 'ICPUSDT', 
        'ATOMUSDT', 'UNIUSDT', 'LINKUSDT', 'AVAXUSDT', 'HYPEUSDT', 'LTCUSDT', 'SOLUSDT', 'AAVEUSDT', 
        'ETHUSDT', 'BTCUSDT'
    ]
    results = []
    for asset in assets:
        res = train_asset_comp(asset)
        if res: results.append(res)
    pd.DataFrame(results).to_csv("/Users/anders/projects/futures/comprehensive_audit_results.csv", index=False)
    print("Comprehensive Audit Complete.")
