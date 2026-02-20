import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# --- Bayesian Linear Layer (Gaussian Weights) ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Reparameterization: w = mu + sigma * eps
        # sigma = log(1 + exp(rho))
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-4.0))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).fill_(-4.0))
        
    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_epsilon = torch.randn_like(weight_sigma)
        weight = self.weight_mu + weight_sigma * weight_epsilon
        
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_epsilon = torch.randn_like(bias_sigma)
        bias = self.bias_mu + bias_sigma * bias_epsilon
        
        return F.linear(x, weight, bias)

# --- Piecewise Constant Activation (Trainable 256 bins) ---
class PWCActivation(nn.Module):
    def __init__(self, num_bins=256, range_min=-5.0, range_max=5.0):
        super(PWCActivation, self).__init__()
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.step = (range_max - range_min) / num_bins
        
        # Trainable bin values (initialized as ReLU-ish or Sigmoid-ish)
        # Using a grid of values that backprop can update
        init_bins = torch.linspace(range_min, range_max, num_bins)
        # Apply a ReLU-like initialization
        init_bins = torch.clamp(init_bins, min=0.0)
        self.bins = nn.Parameter(init_bins)
        
    def forward(self, x):
        # Discretize input to bin indices
        # We use a straight-through estimator or similar for backprop through indices?
        # Actually, if we want the bins to be trainable, we need to know WHICH bin x fell into.
        # index = floor((x - min) / step)
        
        # Hard discretization for PWC
        idx = ((x - self.range_min) / self.step).long()
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        
        # Gather bin values
        # Out: bins[idx]
        # To make it differentiable w.r.t the bins:
        # We can use a one-hot like gather or just indexing which is differentiable in PyTorch for the values.
        out = self.bins[idx]
        
        # Stochasticity: Sample a small perturbation or simulate "stochastic neuron firing"
        # as requested "new random value each time evaluated"
        noise = torch.randn_like(out) * 0.01 
        return out + noise

# --- Hybrid Model ---
class HybridFFNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, output_dim=1):
        super(HybridFFNN, self).__init__()
        self.l1 = BayesianLinear(input_dim, hidden_dim)
        self.act1 = PWCActivation(num_bins=256)
        self.l2 = BayesianLinear(hidden_dim, hidden_dim)
        self.act2 = PWCActivation(num_bins=256)
        self.l3 = BayesianLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        x = self.l3(x)
        return x

# --- 32D Sphere Data Generation ---
def generate_sphere_data(n_samples=10000, dim=32):
    X = torch.randn(n_samples, dim)
    # y = sum(x^2) + noise
    y = torch.sum(X**2, dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1
    return X, y

def train_hybrid_audit(time_limit=120):
    print(">>> Zenith: Piecewise Constant Bayesian Audit (32D Sphere) <<<")
    X, y = generate_sphere_data(20000, 32)
    
    # Split
    train_split = 16000
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:], y[train_split:]
    
    model = HybridFFNN(input_dim=32, hidden_dim=512)
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    
    start_time = time.time()
    epoch = 0
    best_loss = float('inf')
    
    while time.time() - start_time < time_limit:
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if time.time() - start_time > time_limit: break
            
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'zenith_hybrid_best.pth')
            
        if epoch % 5 == 0:
            elapsed = int(time.time() - start_time)
            print(f"  Epoch {epoch:03d} | Val Loss: {val_loss:.6f} | Time: {elapsed}s")
        epoch += 1
        
    print(f"\nAudit complete. Best Val Loss: {best_loss:.6f}")
    
    # Visualization of learned activation curves
    model.load_state_dict(torch.load('zenith_hybrid_best.pth'))
    plot_activation(model.act1, "Activation_Layer_1")
    plot_activation(model.act2, "Activation_Layer_2")

def plot_activation(act_layer, name):
    bins = act_layer.bins.detach().numpy()
    x_axis = np.linspace(act_layer.range_min, act_layer.range_max, act_layer.num_bins)
    
    plt.figure(figsize=(8, 5))
    plt.step(x_axis, bins, where='post', color='cyan', label='Learned PWC Bins')
    plt.title(f"Learned Piecewise Constant Activation: {name}")
    plt.xlabel("Input x")
    plt.ylabel("Output f(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.close()
    print(f"  Saved {name}.png")

if __name__ == "__main__":
    train_hybrid_audit()
