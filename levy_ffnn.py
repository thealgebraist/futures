import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# --- Stable Levy Stable Noise Layer ---
class LevyActivation(nn.Module):
    def __init__(self, alpha=1.5, scale=0.01):
        super(LevyActivation, self).__init__()
        self.alpha = alpha
        self.scale = scale
        
    def forward(self, x):
        if self.training:
            # Chambers-Mallows-Stuck method for symmetric alpha-stable
            # Ensure u is strictly within (-pi/2, pi/2)
            u = (torch.rand_like(x) * 0.9999 - 0.49995) * np.pi
            # Ensure w is strictly positive
            w = -torch.log(torch.rand_like(x) * 0.9999 + 1e-5) 
            
            a = self.alpha
            # S = [sin(a*u) / cos(u)^(1/a)] * [cos((1-a)*u) / w]^((1-a)/a)
            
            t1 = torch.sin(a * u) / (torch.cos(u) ** (1.0/a))
            t2 = (torch.cos((1.0 - a) * u) / w) ** ((1.0 - a) / a)
            
            noise = t1 * t2 * self.scale
            
            # Robust clamp
            noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
            noise = torch.clamp(noise, -1.0, 1.0)
            
            return torch.tanh(x) + noise
        else:
            return torch.tanh(x)

class LevyFFNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32):
        super(LevyFFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            LevyActivation(alpha=1.5, scale=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            LevyActivation(alpha=1.5, scale=0.01),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize weights carefully
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.net(x)

def load_data(symbol, asset_type):
    if asset_type == 'stock':
        path = f'data/signature_experiment/{symbol}_4y.csv'
    else:
        if 'BTC' in symbol: path = 'data/btc_experiment/btc_1m_3mo.csv'
        elif 'SOL' in symbol: path = 'data/sol_experiment/sol_10m_1y.csv'
        else: path = f'data/ecoins_2y/{symbol}USDT_10m_2y.csv'
            
    if not os.path.exists(path):
        return None
        
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values
    return prices

def create_sequences(data, seq_len=16):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def train_and_eval(symbol, asset_type):
    print(f"Processing {symbol}...", flush=True)
    prices = load_data(symbol, asset_type)
    if prices is None or len(prices) < 100: return [1.0], [1.0]
    
    rets = np.diff(prices) / (prices[:-1] + 1e-9)
    mean, std = np.mean(rets), np.std(rets)
    rets = (rets - mean) / (std + 1e-6)
    
    X, y = create_sequences(rets)
    split = int(len(X) * 0.8)
    X_train, X_test = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    y_train, y_test = torch.FloatTensor(y[:split]).view(-1, 1), torch.FloatTensor(y[split:]).view(-1, 1)
    
    model = LevyFFNN()
    optimizer = optim.RAdam(model.parameters(), lr=5e-4) # Lower LR for stability
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < 60:
        model.train()
        indices = torch.randint(0, len(X_train), (32,))
        bx, by = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        
        if torch.isnan(loss):
            print(f"Warning: NaN loss for {symbol} at step {steps}. Restarting optimization.", flush=True)
            model = LevyFFNN()
            optimizer = optim.RAdam(model.parameters(), lr=5e-4)
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if steps % 100 == 0:
            model.eval()
            with torch.no_grad():
                v_loss = criterion(model(X_test), y_test).item()
            train_losses.append(loss.item())
            val_losses.append(v_loss)
        steps += 1
        
    return train_losses, val_losses

def main():
    stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
    ecoins = ['BTC', 'ETH', 'SOL', 'BNB']
    results = {}
    
    for s in stocks: results[s] = train_and_eval(s, 'stock')
    for c in ecoins: results[c] = train_and_eval(c, 'ecoin')

    plt.figure(figsize=(12, 8))
    for name, (tl, vl) in results.items():
        if vl: plt.plot(vl, label=f'{name} Val')
    plt.title('Levy FFNN: Stable Validation Error Curves')
    plt.yscale('log')
    plt.legend()
    plt.savefig('levy_error_curves.png')
    
    with open('levy_stats.txt', 'w') as f:
        for name, (tl, vl) in results.items():
            f.write(f"{name} Final Val MSE: {vl[-1] if vl else 0.0:.6f}\n")

if __name__ == "__main__":
    main()
