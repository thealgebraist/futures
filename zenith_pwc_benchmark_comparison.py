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
    def __init__(self, num_bins=256, range_min=-5.0, range_max=5.0, init_mode='standard'):
        super(PWCActivation, self).__init__()
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.step = (range_max - range_min) / num_bins
        
        if init_mode == 'standard':
            # ReLU-like ramp initialization
            init_bins = torch.linspace(range_min, range_max, num_bins)
            init_bins = torch.clamp(init_bins, min=0.0)
        else:
            # Random initialization N(0, 1)
            init_bins = torch.randn(num_bins) * 0.5
            
        self.bins = nn.Parameter(init_bins)
        
    def forward(self, x):
        idx = ((x - self.range_min) / self.step).long()
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        out = self.bins[idx]
        # User asked for stochastic inference "random value each time evaluated"
        noise = torch.randn_like(out) * 0.01 
        return out + noise

# --- Hybrid Model ---
class HybridFFNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, output_dim=1, init_mode='standard'):
        super(HybridFFNN, self).__init__()
        self.l1 = BayesianLinear(input_dim, hidden_dim)
        self.act1 = PWCActivation(num_bins=256, init_mode=init_mode)
        self.l2 = BayesianLinear(hidden_dim, hidden_dim)
        self.act2 = PWCActivation(num_bins=256, init_mode=init_mode)
        self.l3 = BayesianLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        return self.l3(x)

# --- Vanilla Model ---
class VanillaFFNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, output_dim=1):
        super(VanillaFFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 32D Sphere Data Generation ---
def generate_sphere_data(n_samples=25000, dim=32):
    X = torch.randn(n_samples, dim)
    y = torch.sum(X**2, dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1
    return X, y

def train_model(model_instance, model_name, X, y, time_limit=120):
    print(f"  Training {model_name}...")
    train_split = 20000
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:], y[train_split:]
    
    optimizer = optim.RAdam(model_instance.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    
    loss_history = []
    start_time = time.time()
    epoch = 0
    
    while time.time() - start_time < time_limit:
        model_instance.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            output = model_instance(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            if time.time() - start_time > time_limit: break
            
        model_instance.eval()
        with torch.no_grad():
            val_loss = criterion(model_instance(X_val), y_val).item()
        
        elapsed = time.time() - start_time
        loss_history.append((elapsed, val_loss))
        
        if epoch % 10 == 0:
            print(f"    {model_name} | Epoch {epoch:03d} | Val Loss: {val_loss:.4f} | Time: {int(elapsed)}s")
        epoch += 1
        
    return loss_history

def run_benchmark():
    print(">>> Zenith: Hybrid PWC Initialization Benchmark (32D Sphere) <<<")
    X, y = generate_sphere_data()
    
    # 1. Vanilla Baseline
    vanilla_model = VanillaFFNN(32, 512, 1)
    vanilla_history = train_model(vanilla_model, "Vanilla_FFNN", X, y, time_limit=120)
    
    # 2. Hybrid Standard (ReLU Ramp)
    hybrid_std_model = HybridFFNN(32, 512, 1, init_mode='standard')
    hybrid_std_history = train_model(hybrid_std_model, "Hybrid_Standard_Init", X, y, time_limit=120)
    # Save curves for Standard Init
    plot_activation(hybrid_std_model.act1, "Activation_Standard_Init_L1")
    plot_activation(hybrid_std_model.act2, "Activation_Standard_Init_L2")
    
    # 3. Hybrid Random (Gaussian Noise)
    hybrid_rnd_model = HybridFFNN(32, 512, 1, init_mode='random')
    hybrid_rnd_history = train_model(hybrid_rnd_model, "Hybrid_Random_Init", X, y, time_limit=120)
    # Save curves for Random Init
    plot_activation(hybrid_rnd_model.act1, "Activation_Random_Init_L1")
    plot_activation(hybrid_rnd_model.act2, "Activation_Random_Init_L2")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    v_time, v_loss = zip(*vanilla_history)
    hs_time, hs_loss = zip(*hybrid_std_history)
    hr_time, hr_loss = zip(*hybrid_rnd_history)
    
    plt.plot(v_time, v_loss, label='Vanilla FFNN (ReLU)', color='magenta', linewidth=1.5, alpha=0.8)
    plt.plot(hs_time, hs_loss, label='Hybrid PWC-Bayesian (Std Init)', color='cyan', linewidth=2)
    plt.plot(hr_time, hr_loss, label='Hybrid PWC-Bayesian (Random Init)', color='orange', linewidth=2, linestyle='--')
    
    plt.yscale('log')
    plt.title("Initialization Benchmark: PWC-Bayesian vs Vanilla FFNN")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Validation MSE (Log Scale)")
    plt.grid(True, alpha=0.3, which="both", ls="--")
    plt.legend()
    plt.savefig("zenith_init_benchmark.png")
    
    print("\nBenchmark Complete. Plot saved as zenith_init_benchmark.png")
    print(f"Final Vanilla Loss: {v_loss[-1]:.6f}")
    print(f"Final Hybrid Std Loss: {hs_loss[-1]:.6f}")
    print(f"Final Hybrid Random Loss: {hr_loss[-1]:.6f}")

def plot_activation(act_layer, name):
    bins = act_layer.bins.detach().numpy()
    x_axis = np.linspace(act_layer.range_min, act_layer.range_max, act_layer.num_bins)
    plt.figure(figsize=(8, 5))
    plt.step(x_axis, bins, where='post', color='orange')
    plt.title(f"Learned PWC Activation: {name}")
    plt.xlabel("Input x")
    plt.ylabel("Output f(x)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{name}.png")
    plt.close()

if __name__ == "__main__":
    run_benchmark()
