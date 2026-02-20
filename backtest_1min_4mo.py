import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Model definition
class FFNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=512, output_dim=4):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Load model
model = FFNN()
model.load_state_dict(torch.load('predictive_ffnn.pth'))
model.eval()

# Load data
df = pd.read_csv('data/futures_1m_4mo.csv')
target_col = 'NQ=F'
values = df[target_col].values
values = values[~np.isnan(values)]

split_idx = int(0.8 * len(values))
test_values = values[split_idx:]

# Scaler
scaler = StandardScaler()
values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
test_values_scaled = values_scaled[split_idx:]

# Simulation Parameters
initial_capital = 100.0
threshold = 0.0001
input_steps = 16

# Vectorized simulation
def run_vectorized_sim():
    num_steps = len(test_values_scaled) - input_steps - 1
    print(f"Preparing vectorized windows for {num_steps} steps...")
    
    # Create all windows at once
    # This might consume more memory but is much faster
    X_all = []
    for i in range(num_steps):
        X_all.append(test_values_scaled[i : i + input_steps])
    X_all = torch.FloatTensor(np.array(X_all))
    
    print("Running batch inference...")
    with torch.no_grad():
        Y_pred_all = model(X_all).numpy()
    
    # Predicted change: avg of future 4 - last input price
    last_prices = test_values_scaled[input_steps - 1 : input_steps - 1 + num_steps]
    predicted_avgs = np.mean(Y_pred_all, axis=1)
    predicted_changes = predicted_avgs - last_prices
    
    # Calculate returns
    price_today = test_values[input_steps - 1 : input_steps - 1 + num_steps]
    price_next = test_values[input_steps : input_steps + num_steps]
    actual_returns = (price_next - price_today) / price_today
    
    # Strategy: Bidirectional
    pos_bi = np.zeros(num_steps)
    pos_bi[predicted_changes > threshold] = 1
    pos_bi[predicted_changes < -threshold] = -1
    
    # Strategy: Short-only
    pos_short = np.zeros(num_steps)
    pos_short[predicted_changes < -threshold] = -1
    
    # Calculate equity curves
    # V_{t+1} = V_t * (1 + r_t * p_t)
    equity_bi = [initial_capital]
    equity_short = [initial_capital]
    
    current_bi = initial_capital
    current_short = initial_capital
    
    for r, pb, ps in zip(actual_returns, pos_bi, pos_short):
        current_bi *= (1 + r * pb)
        current_short *= (1 + r * ps)
        equity_bi.append(current_bi)
        equity_short.append(current_short)
        
    return equity_bi, equity_short

equity_bi, equity_short = run_vectorized_sim()

print(f"Results for 1min (4 months):")
print(f"Bidirectional Final Value: ${equity_bi[-1]:.2f}")
print(f"Short-Only Final Value: ${equity_short[-1]:.2f}")

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(equity_short, label='Short-Only (1m)', color='red')
plt.plot(equity_bi, label='Bidirectional (1m)', color='green', alpha=0.5)
plt.axhline(100, color='black', linestyle='--', alpha=0.3)
plt.title('Performance Comparison: 1min BTCUSDT (4 Months)')
plt.ylabel('Value ($)')
plt.xlabel('Steps (minutes)')
plt.legend()
plt.grid(True)
plt.savefig('performance_1min_4mo.png')

# Save stats
results = pd.DataFrame({
    'equity_short': equity_short,
    'equity_bi': equity_bi
})
results.to_csv('results_1min_4mo.csv', index=False)
