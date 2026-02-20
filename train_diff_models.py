import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Neural ODE Implementation
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, t, h):
        return self.net(h)

class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralODE, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.t = torch.tensor([0, 1]).float()
        
    def forward(self, x):
        h0 = self.input_layer(x)
        h_traj = odeint(self.ode_func, h0, self.t.to(device))
        hT = h_traj[-1]
        return self.output_layer(hT)

# 2. Liquid Time-Constant (LTC) Simplified Formulation
class LTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LTCModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.A = nn.Parameter(torch.ones(hidden_dim))
        self.f_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.size()
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        dt = 0.1
        for t in range(seq_len):
            x = x_seq[:, t, :]
            inp = torch.cat([x, h], dim=1)
            f = self.f_net(inp)
            dh = (- (1.0/torch.exp(self.tau) + f) * h + f * self.A) * dt
            h = h + dh
        return self.output_layer(h)

def train_and_eval(model, train_loader, X_test, y_test, scaler_y, name="Model", duration=120):
    optimizer = optim.RAdam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    start_time = time.time()
    
    print(f"Training {name}...", flush=True)
    epoch = 0
    while time.time() - start_time < duration:
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        epoch += 1
        if epoch % 5 == 0:
            print(f"{name} Epoch {epoch} complete", flush=True)
        
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_test).to(device)
        preds_scaled = model(test_inputs).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled)
        mse = np.mean((preds.flatten() - y_test.flatten())**2)
        
    print(f"{name} Final MSE: {mse:.6f}", flush=True)
    return mse

def main():
    if not os.path.exists('data/futures_10m_v2.csv'):
        print("Data not found.")
        return
        
    df = pd.read_csv('data/futures_10m_v2.csv', index_col=0)
    data = df.values
    lookback = 10
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 5])
        
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, 8)
    X_test_reshaped = X_test.reshape(-1, 8)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(-1, lookback, 8)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(-1, lookback, 8)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_train_ode = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_test_ode = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
    
    train_ds_ode = TensorDataset(torch.FloatTensor(X_train_ode), torch.FloatTensor(y_train_scaled))
    train_loader_ode = DataLoader(train_ds_ode, batch_size=64, shuffle=True)
    
    train_ds_ltc = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    train_loader_ltc = DataLoader(train_ds_ltc, batch_size=64, shuffle=True)
    
    ode_model = NeuralODE(80, 128, 1).to(device)
    mse_ode = train_and_eval(ode_model, train_loader_ode, X_test_ode, y_test, scaler_y, name="NeuralODE")
    
    ltc_model = LTCModel(8, 64, 1).to(device)
    mse_ltc = train_and_eval(ltc_model, train_loader_ltc, X_test_scaled, y_test, scaler_y, name="LTC")
    
    with open("diff_results.txt", "w") as f:
        f.write(f"NeuralODE_MSE: {mse_ode}\n")
        f.write(f"LTC_MSE: {mse_ltc}\n")

if __name__ == "__main__":
    main()
