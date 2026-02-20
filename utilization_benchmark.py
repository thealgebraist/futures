import torch
import torch.nn as nn
import numpy as np
import time
from scipy.optimize import linprog

# FFNN Predictor (uP scaled architecture)
class UtilizationPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1) # Output: Predicted Available Capacity
        )
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def benchmark_ffnn(iters=1000):
    model = UtilizationPredictor()
    model.eval()
    x = torch.randn(1, 5)
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    end = time.perf_counter()
    avg_latency = (end - start) / iters
    return avg_latency * 1000 # in ms

def benchmark_milp(n_tasks=10, iters=100):
    # Objective: Maximize rewards
    c = -np.random.rand(n_tasks)
    
    # Constraints: CPU, GPU, RAM usage
    A = np.random.rand(3, n_tasks)
    b = np.array([0.24, 0.24, 0.24]) # Available capacity (99% - 75%)
    
    start = time.perf_counter()
    for _ in range(iters):
        res = linprog(c, A_ub=A, b_ub=b, method='highs')
    end = time.perf_counter()
    
    avg_latency = (end - start) / iters
    return avg_latency * 1000 # in ms

if __name__ == "__main__":
    print("Running Server Utilization Benchmarks...")
    ffnn_latency = benchmark_ffnn()
    print(f"FFNN Prediction Latency: {ffnn_latency:.4f} ms")
    
    milp_latency = benchmark_milp()
    print(f"MILP Scheduling Latency: {milp_latency:.4f} ms")
    
    with open("utilization_benchmarks.txt", "w") as f:
        f.write(f"FFNN_Latency_ms: {ffnn_latency:.4f}\n")
        f.write(f"MILP_Latency_ms: {milp_latency:.4f}\n")
