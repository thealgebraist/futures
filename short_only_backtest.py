import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
df = pd.read_csv('data/futures_10m.csv')
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

def run_simulation(strategy_type='short_only'):
    capital = initial_capital
    equity = [capital]
    positions = []
    
    for i in range(len(test_values_scaled) - input_steps - 1):
        x_input = torch.FloatTensor(test_values_scaled[i : i + input_steps]).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x_input).numpy().flatten()
            
        last_price_scaled = test_values_scaled[i + input_steps - 1]
        predicted_change = np.mean(y_pred) - last_price_scaled
        
        pos = 0
        if strategy_type == 'short_only':
            if predicted_change < -threshold:
                pos = -1
        elif strategy_type == 'bidirectional':
            if predicted_change > threshold:
                pos = 1
            elif predicted_change < -threshold:
                pos = -1
                
        # Calculate return
        ret = (test_values[i + input_steps] - test_values[i + input_steps - 1]) / test_values[i + input_steps - 1]
        capital *= (1 + ret * pos)
        equity.append(capital)
        positions.append(pos)
        
    return equity, positions

# Run both for comparison
equity_short, pos_short = run_simulation('short_only')
equity_bi, pos_bi = run_simulation('bidirectional')

print(f"Short-Only Results:")
print(f"Final Value: ${equity_short[-1]:.2f}")
print(f"Total Return: {(equity_short[-1] - 100):.2f}%")

# Save Comparison Plot
plt.figure(figsize=(10, 8))
plt.plot(equity_short, label='Short-Only Equity ($)', color='red')
plt.plot(equity_bi, label='Bidirectional Equity ($)', color='green', alpha=0.5)
plt.axhline(100, color='black', linestyle='--', alpha=0.3)
plt.title('Short-Only vs Bidirectional Performance ($100 Init)')
plt.ylabel('Value ($)')
plt.xlabel('Steps')
plt.legend()
plt.grid(True)
plt.savefig('short_only_comparison.png')

# Save stats
results = pd.DataFrame({
    'short_only_equity': equity_short,
    'bidirectional_equity': equity_bi
})
results.to_csv('short_only_results.csv', index=False)
