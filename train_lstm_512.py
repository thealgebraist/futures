import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def simulate_trading(y_true, y_pred, initial_capital=100.0, contract_multiplier=5.0):
    capital = initial_capital
    history = [capital]
    
    for i in range(1, len(y_pred)):
        pred_change = y_pred[i] - y_true[i-1]
        actual_change = y_true[i] - y_true[i-1]
        
        if pred_change > 0:
            pnl = actual_change * contract_multiplier - 1.50
            capital += pnl
        elif pred_change < 0:
            pnl = -actual_change * contract_multiplier - 1.50
            capital += pnl
            
        history.append(capital)
        if capital <= 0:
            capital = 0
            break
            
    return capital, history

def main():
    if not os.path.exists('data/futures_10m_v2.csv'):
        print("Data file not found.")
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
    
    X_train_reshaped = X_train.reshape(-1, 8)
    X_test_reshaped = X_test.reshape(-1, 8)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(-1, lookback, 8)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(-1, lookback, 8)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = LSTMModel(8, 512, 2).to(device)
    optimizer = optim.RAdam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    print("Training 512-neuron LSTM on 8 features...", flush=True)
    start_time = time.time()
    while time.time() - start_time < 120:
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
    print("Training complete. Starting simulation...", flush=True)
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_test_scaled).to(device)
        preds_scaled = model(test_inputs).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled)
        
    final_cap, cap_history = simulate_trading(y_test.flatten(), preds.flatten())
    print(f"Initial Capital: $100.00", flush=True)
    print(f"Final Capital after 1 month unseen data: ${final_cap:.2f}", flush=True)
    
    with open("sim_results.txt", "w") as f:
        f.write(f"Final_Capital: {final_cap}\n")
        f.write(f"History: {cap_history[-20:]}\n")

if __name__ == "__main__":
    main()
