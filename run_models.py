import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Piecewise Linear Activation with 128 segments
class PiecewiseLinearActivation(nn.Module):
    def __init__(self, num_segments=128, range_min=-5, range_max=5):
        super(PiecewiseLinearActivation, self).__init__()
        self.num_segments = num_segments
        self.grid = nn.Parameter(torch.linspace(range_min, range_max, num_segments + 1), requires_grad=False)
        self.slopes = nn.Parameter(torch.ones(num_segments))
        
    def forward(self, x):
        out = torch.zeros_like(x)
        for i in range(self.num_segments):
            threshold = self.grid[i]
            out += self.slopes[i] * torch.relu(x - threshold)
        return out

# Model 1: 32-neuron FFNN
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Model 2: 32-neuron FFNN with 128 PWL
class FFNN_PWL(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(FFNN_PWL, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.pwl = PiecewiseLinearActivation(num_segments=128)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.pwl(out)
        out = self.fc2(out)
        return out

# Model 4: LSTM (as per TCM-ABC-LSTM paper)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_torch_model(model, train_loader, test_loader, duration=120, name="Model"):
    optimizer = optim.RAdam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    start_time = time.time()
    
    train_losses = []
    test_losses = []
    
    epoch = 0
    while time.time() - start_time < duration:
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_losses.append(np.mean(batch_losses))
        
        model.eval()
        with torch.no_grad():
            t_losses = []
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                t_loss = criterion(outputs, targets)
                t_losses.append(t_loss.item())
            test_losses.append(np.mean(t_losses))
        
        epoch += 1
        if epoch % 10 == 0:
            print(f"{name} Epoch {epoch}, Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}", flush=True)
            
    return train_losses, test_losses

def main():
    df = pd.read_csv('data/futures_10m.csv', index_col=0)
    data = df.values
    
    lookback = 10
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback].flatten())
        y.append(data[i+lookback, 4]) # Target MES=F (column 4)
        
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    print("Training Model 1: FFNN 32...")
    model1 = FFNN(X_train.shape[1], 32).to(device)
    h1_train, h1_test = train_torch_model(model1, train_loader, test_loader, name="FFNN32")
    
    print("Training Model 2: FFNN 32 PWL...")
    model2 = FFNN_PWL(X_train.shape[1], 32).to(device)
    h2_train, h2_test = train_torch_model(model2, train_loader, test_loader, name="FFNN32_PWL")
    
    print("Training Model 3: Gaussian Process...")
    gp_subset = 1000
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    gp.fit(X_train[:gp_subset], y_train[:gp_subset])
    y_pred_gp = gp.predict(X_test)
    gp_test_loss = np.mean((y_pred_gp - y_test.flatten())**2)
    print(f"GP Test Loss: {gp_test_loss:.6f}")
    
    print("Training Model 4: LSTM...")
    X_train_lstm = X_train.reshape(-1, lookback, 5)
    X_test_lstm = X_test.reshape(-1, lookback, 5)
    train_ds_lstm = TensorDataset(torch.FloatTensor(X_train_lstm), torch.FloatTensor(y_train))
    test_ds_lstm = TensorDataset(torch.FloatTensor(X_test_lstm), torch.FloatTensor(y_test))
    train_loader_lstm = DataLoader(train_ds_lstm, batch_size=32, shuffle=True)
    test_loader_lstm = DataLoader(test_ds_lstm, batch_size=32)
    
    model4 = LSTMModel(5, 100, 2).to(device)
    h4_train, h4_test = train_torch_model(model4, train_loader_lstm, test_loader_lstm, name="LSTM")
    
    plt.figure(figsize=(10, 6))
    plt.plot(h1_test, label='FFNN 32')
    plt.plot(h2_test, label='FFNN 32 PWL')
    plt.plot(h4_test, label='LSTM')
    plt.axhline(y=gp_test_loss, color='r', linestyle='--', label='GP')
    plt.yscale('log')
    plt.title('Model Comparison')
    plt.legend()
    plt.savefig('error_curves.png')
    
    with open("results.txt", "w") as f:
        f.write(f"FFNN32: {h1_test[-1]}\n")
        f.write(f"FFNN32_PWL: {h2_test[-1]}\n")
        f.write(f"GP: {gp_test_loss}\n")
        f.write(f"LSTM: {h4_test[-1]}\n")

if __name__ == "__main__":
    main()
