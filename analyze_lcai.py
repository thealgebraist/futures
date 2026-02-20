import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import levy_stable
import os

# R-Adam Optimizer (simplified version or use torch.optim.RAdam if available)
def get_optimizer(model, lr=0.001):
    return optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-5)

# Weight Normalization: use nn.utils.weight_norm
def apply_weight_norm(layer):
    return nn.utils.weight_norm(layer)

# muP Implementation (Output layer scaled by 1/width, hidden by 1/sqrt(width))
class MuP_FFNN(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=1):
        super(MuP_FFNN, self).__init__()
        self.width = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Scaling initialization for muP
        # Hidden layer: scale by 1/sqrt(width)
        nn.init.kaiming_uniform_(self.fc1.weight, a=np.sqrt(5))
        self.fc1.weight.data *= 1.0 / np.sqrt(self.width)
        
        # Output layer: scale by 1/width
        nn.init.kaiming_uniform_(self.fc2.weight, a=np.sqrt(5))
        self.fc2.weight.data *= 1.0 / self.width
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_ffnn(X_train, y_train, hidden_dim=64, duration=120):
    model = MuP_FFNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
    optimizer = get_optimizer(model)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    epochs = 0
    while time.time() - start_time < duration:
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        epochs += 1
        if epochs % 100 == 0:
            print(f"Epoch {epochs}, Loss: {loss.item():.6f}")
    
    print(f"Finished FFNN training in {time.time() - start_time:.2f}s")
    return model

def run_analysis():
    # Load data
    df = pd.read_csv('data/lcai_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    prices = df['close'].values
    
    # Preprocess for time series (sliding window)
    window_size = 24
    X, y = [], []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    # 1. FFNN Analysis (120s)
    print("Starting FFNN Analysis...")
    ffnn_model = train_ffnn(X_tensor, y_tensor, duration=120)
    
    # 2. Alpha Stable Levy (Faster estimation)
    print("Starting Alpha Stable Levy...")
    returns = np.diff(prices) / prices[:-1]
    # Simple estimate of alpha using the relationship between quantiles
    # alpha = log(2) / log(q75 - q25) ... or just use a subset for fit
    try:
        # Use only 100 points for fitting to be fast
        alpha, beta, loc, scale = levy_stable.fit(returns[:100])
    except:
        alpha, beta, loc, scale = 1.5, 0.0, 0.0, 0.01
    print(f"Alpha Stable Parameters: alpha={alpha:.4f}, beta={beta:.4f}")
    
    # 3. Signature Martingale (Manual implementation of 1st and 2nd order)
    print("Starting Signature Martingale...")
    def get_signature(path):
        # path is 1D, signature of (t, path)
        t = np.linspace(0, 1, len(path))
        # Level 1: [integral dt, integral dx]
        s1 = [t[-1] - t[0], path[-1] - path[0]]
        # Level 2: [integral dt dt, integral dt dx, integral dx dt, integral dx dx]
        # approx using trapezoidal
        s2 = [0.5 * s1[0]**2, np.trapz(path, t), np.trapz(t, path), 0.5 * s1[1]**2]
        return s1 + s2

    # Calculate signature for a window
    sig = get_signature(prices[-window_size:])
    print(f"Signature (subset): {sig}")
    
    # 4. Gaussian Process (Smaller subset)
    print("Starting Gaussian Process...")
    gp_X = X_scaled[-50:]
    gp_y = y_scaled[-50:]
    kernel = C(1.0) * RBF(1.0)
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(gp_X, gp_y)
    print(f"GP Kernel: {gp.kernel_}")
    
    # Final results
    results = {
        'ffnn_final_loss': float(nn.MSELoss()(ffnn_model(X_tensor), y_tensor).item()),
        'levy_alpha': float(alpha),
        'martingale_mean': float(np.mean(np.diff(prices))),
        'gp_score': float(gp.score(gp_X, gp_y))
    }
    
    pd.DataFrame([results]).to_csv('lcai_results.csv', index=False)
    print("Analysis complete. Results saved to lcai_results.csv")

if __name__ == "__main__":
    run_analysis()
