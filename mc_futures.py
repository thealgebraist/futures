import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_paths(N, T, dt):
    """
    Simulates N paths using a Sum of Gaussian Walks framework.
    """
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps)
    paths = np.zeros((N, num_steps))
    
    # Parameters
    # 1. Gaussian Walks (Multi-scale noise)
    sigma_fast = 0.02
    sigma_slow = 0.005
    
    # 2. Piecewise components (Jump at T/2)
    t_split = T / 2
    mu1, mu2 = 0.01, -0.02
    vol_scale1, vol_scale2 = 1.0, 2.5 # Second-order jump (acceleration)
    
    # 3. Cyclical
    A, freq, phase = 0.05, 2 * np.pi, 0
    
    # 4. Dirac Jumps (Poisson trigger)
    jump_prob = 0.01 # 1% per step
    jump_size = 0.05
    
    # Pre-calculate noise for all paths and steps
    # Combined noise: vol_scale(t) * (sigma_fast * eps_f + sigma_slow * eps_s)
    eps_fast = np.random.normal(0, np.sqrt(dt), (N, num_steps))
    eps_slow = np.random.normal(0, np.sqrt(dt), (N, num_steps))
    
    # Iterate through time steps
    current_price = np.zeros(N)
    for i in range(num_steps):
        time = t[i]
        
        # Piecewise selection
        if time < t_split:
            mu = mu1
            vol_scale = vol_scale1
        else:
            mu = mu2
            vol_scale = vol_scale2
            
        # Increments
        # Gaussian components
        dw = vol_scale * (sigma_fast * eps_fast[:, i] + sigma_slow * eps_slow[:, i])
        
        # Drift component
        ddrift = mu * dt
        
        # Cyclical component (Derivative of Sine)
        dcyclic = A * freq * np.cos(freq * time + phase) * dt
        
        # Dirac shocks
        djumps = np.where(np.random.rand(N) < jump_prob, jump_size, 0)
        
        # Update price
        current_price += dw + ddrift + dcyclic + djumps
        paths[:, i] = current_price
        
    return t, paths

# Run Simulation
N = 10000
T = 1.0
dt = 0.01
t, paths = simulate_paths(N, T, dt)

# Stats
terminal_values = paths[:, -1]
expected_value = np.mean(terminal_values)
volatility = np.std(terminal_values)
confidence_interval = (expected_value - 1.96 * volatility / np.sqrt(N), 
                       expected_value + 1.96 * volatility / np.sqrt(N))

print(f"Monte Carlo Results (N={N}):")
print(f"Expected Terminal Value: {expected_value:.6f}")
print(f"Standard Deviation: {volatility:.6f}")
print(f"95% Confidence Interval: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")

# Save results
df = pd.DataFrame(terminal_values, columns=['terminal_value'])
df.to_csv('simulation_results.csv', index=False)

# Plotting
plt.figure(figsize=(12, 10))

# Subplot 1: Representative Paths
plt.subplot(2, 1, 1)
for i in range(10): # Plot 10 paths
    plt.plot(t, paths[i, :], alpha=0.7)
plt.axvline(x=T/2, color='r', linestyle='--', label='Regime Switch (T/2)')
plt.title(f"FX Future Paths: Sum of Gaussian Walks (N={N})")
plt.xlabel("Time (T)")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Terminal Value Distribution
plt.subplot(2, 1, 2)
plt.hist(terminal_values, bins=100, density=True, color='skyblue', alpha=0.7, ec='black')
plt.axvline(expected_value, color='red', linestyle='solid', linewidth=2, label=f'E[S(T)] = {expected_value:.4f}')
plt.title("Distribution of Terminal Values")
plt.xlabel("Terminal Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mc_visualization.png')
plt.show() # Note: In headful this might block, but here it's likely handled by the wrapper.
