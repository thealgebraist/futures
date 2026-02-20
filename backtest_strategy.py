import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Model definition (must match training)
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

# Load and prepare data
df = pd.read_csv('data/futures_10m.csv')
target_col = 'NQ=F'
values = df[target_col].values
values = values[~np.isnan(values)]

split_idx = int(0.8 * len(values))
test_values = values[split_idx:]

# We need the same scaler used during training
# For this minimal version, we re-fit on the whole dataset to get close enough normalization
scaler = StandardScaler()
values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
test_values_scaled = values_scaled[split_idx:]

# Trading simulation
# Parameters
initial_capital = 100.0
capital = initial_capital
equity_curve = [capital]
positions = []
input_steps = 16
output_steps = 4
threshold = 0.0001 # Minimum predicted move to take a position

# Simulate over the test set (roughly 14-20 days)
# One month of data is 4320 steps. If test set is shorter, we report on available data.
num_simulation_steps = len(test_values_scaled) - input_steps - 1
print(f"Starting simulation over {num_simulation_steps} steps...")

for i in range(num_simulation_steps):
    # Current window
    x_input = torch.FloatTensor(test_values_scaled[i : i + input_steps]).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        y_pred = model(x_input).numpy().flatten()
    
    # Simple signal: average predicted direction
    last_price_scaled = test_values_scaled[i + input_steps - 1]
    predicted_avg_future_scaled = np.mean(y_pred)
    predicted_change = predicted_avg_future_scaled - last_price_scaled
    
    # Decision
    if predicted_change > threshold:
        pos = 1 # Long
    elif predicted_change < -threshold:
        pos = -1 # Short
    else:
        pos = 0 # Cash
        
    # Calculate returns (from real prices)
    current_price = test_values[i + input_steps - 1]
    next_price = test_values[i + input_steps]
    actual_return = (next_price - current_price) / current_price
    
    # Update capital
    capital *= (1 + actual_return * pos)
    equity_curve.append(capital)
    positions.append(pos)

# Performance Summary
final_value = capital
total_return = (final_value - initial_capital) / initial_capital * 100
sharpe_ratio = np.mean(np.diff(equity_curve)) / np.std(np.diff(equity_curve)) * np.sqrt(243*365/10) # rough annualized

print(f"Simulation Recap:")
print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Num Steps: {len(equity_curve)}")

# Save data
results_df = pd.DataFrame({
    'equity': equity_curve,
})
results_df.to_csv('backtest_results.csv', index=False)

# Plotting
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(equity_curve, label='Portfolio Value ($)', color='green')
plt.axhline(100, color='red', linestyle='--', alpha=0.5)
plt.title('Portfolio Equity Curve (Initial $100)')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(positions, label='Position (1:Long, -1:Short, 0:Cash)', color='blue', alpha=0.3)
plt.title('Model Positions Over Time')
plt.xlabel('Steps (10 min)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('equity_curve.png')
print("Equity curve saved as equity_curve.png")
