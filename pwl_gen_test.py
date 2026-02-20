import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. Custom PWL Activation Function
class PWLActivation(nn.Module):
    def __init__(self, num_pieces=4):
        super(PWLActivation, self).__init__()
        self.num_pieces = num_pieces
        self.knots = nn.Parameter(torch.linspace(-2, 2, num_pieces - 1))
        self.slopes = nn.Parameter(torch.randn(num_pieces))
        self.intercept = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # f(x) = intercept + slope_0 * x + sum( slope_diff_i * relu(x - knot_i) )
        out = self.intercept + self.slopes[0] * x
        for i in range(self.num_pieces - 1):
            out += (self.slopes[i+1] - self.slopes[i]) * torch.relu(x - self.knots[i])
        return out

# 2. 2-Neuron FFNN
class SmallFFNN(nn.Module):
    def __init__(self, num_pieces=4):
        super(SmallFFNN, self).__init__()
        self.fc1 = nn.Linear(16, 2) # 2 neurons
        self.pwl = PWLActivation(num_pieces)
        self.fc2 = nn.Linear(2, 1) # Predicting next 1 value for simplicity
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.pwl(x)
        # Note: applying pwl to the 2-dim output element-wise
        x = self.fc2(x)
        return x

# 3. Data Preparation
def get_data():
    df = pd.read_csv('data/futures_10m_3mo.csv')
    df.columns = ['Datetime', 'NQ=F']
    values = df['NQ=F'].values
    
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    
    input_steps = 16
    X, y = [], []
    for i in range(len(values_scaled) - input_steps):
        X.append(values_scaled[i : i + input_steps])
        y.append(values_scaled[i + input_steps])
        
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler

# 4. Training and Testing
def train_and_test(m_pieces=4):
    X_train, X_test, y_train, y_test, scaler = get_data()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)
    
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    model = SmallFFNN(num_pieces=m_pieces)
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Training for 4s with M={m_pieces} pieces...")
    start_time = time.time()
    epochs = 0
    while time.time() - start_time < 4.0:
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        epochs += 1
    
    model.eval()
    with torch.no_grad():
        # Test 1: Standard Hold-out
        loss1 = criterion(model(X_test_t), y_test_t).item()
        
        # Test 2: Noise Robustness
        X_noise = X_test_t + 0.1 * torch.randn_like(X_test_t)
        loss2 = criterion(model(X_noise), y_test_t).item()
        
        # Test 3: Distribution Shift (Mean Shift)
        X_shift = X_test_t + 0.5
        loss3 = criterion(model(X_shift), y_test_t).item()
        
        # Test 4: Extrapolation (Extreme scaling)
        X_extrap = X_test_t * 2.0
        loss4 = criterion(model(X_extrap), y_test_t).item()
        
    return [loss1, loss2, loss3, loss4]

if __name__ == "__main__":
    results = {}
    for m in [2, 4, 8, 16]:
        results[m] = train_and_test(m)
        print(f"M={m}: {results[m]}")
        
    # Plotting
    ms = list(results.keys())
    t1 = [r[0] for r in results.values()]
    t2 = [r[1] for r in results.values()]
    t3 = [r[2] for r in results.values()]
    t4 = [r[3] for r in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ms, t1, 'o-', label='Std Hold-out')
    plt.plot(ms, t2, 's-', label='Noise (0.1)')
    plt.plot(ms, t3, '^-', label='Shift (+0.5)')
    plt.plot(ms, t4, 'x-', label='Extrap (x2.0)')
    plt.yscale('log')
    plt.xlabel('Number of PWL Pieces (M)')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('2-Neuron FFNN Generalization Benchmarks')
    plt.legend()
    plt.grid(True)
    plt.savefig('pwl_generalization_results.png')
    print("Results saved to pwl_generalization_results.png")
