import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. Data Preparation (Same as before)
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

# 2. GP Training and Testing
def gp_test():
    X_train, X_test, y_train, y_test, scaler = get_data()
    
    # Kernel: Constant * RBF
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    
    # Using a subset for training if 4s is too short for full GP on 10k points
    # Scikit-learn GP is O(N^3). 12k points might be slow.
    # I'll use a representative subset of 2000 points to ensure it fits in 4s.
    train_subset_size = 1000 
    indices = np.random.choice(len(X_train), train_subset_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-2)
    
    print(f"Training GP for 4s (Subset N={train_subset_size})...")
    start_time = time.time()
    gp.fit(X_train_sub, y_train_sub)
    fit_time = time.time() - start_time
    print(f"GP fit completed in {fit_time:.2f}s.")
    
    # 4 Generalization Tests
    print("Running generalization tests...")
    
    # Test 1: Standard Hold-out
    y_pred1 = gp.predict(X_test)
    loss1 = np.mean((y_pred1 - y_test)**2)
    
    # Test 2: Noise Robustness
    X_noise = X_test + 0.1 * np.random.randn(*X_test.shape)
    y_pred2 = gp.predict(X_noise)
    loss2 = np.mean((y_pred2 - y_test)**2)
    
    # Test 3: Distribution Shift (Mean Shift)
    X_shift = X_test + 0.5
    y_pred3 = gp.predict(X_shift)
    loss3 = np.mean((y_pred3 - y_test)**2)
    
    # Test 4: Extrapolation (Extreme scaling)
    X_extrap = X_test * 2.0
    y_pred4 = gp.predict(X_extrap)
    loss4 = np.mean((y_pred4 - y_test)**2)
    
    return [loss1, loss2, loss3, loss4]

if __name__ == "__main__":
    gp_results = gp_test()
    print(f"GP Results: {gp_results}")
    
    # Comparison with best PWL (M=2) from last run
    # M=2 losses were: [0.0005, 0.0095, 0.2435, 1.0811]
    pwl_m2 = [0.0005, 0.0095, 0.2435, 1.0811]
    
    labels = ['Hold-out', 'Noise', 'Shift', 'Extrap']
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, pwl_m2, width, label='PWL FFNN (M=2)', color='blue', alpha=0.7)
    ax.bar(x + width/2, gp_results, width, label='Gaussian Process', color='orange', alpha=0.7)
    
    ax.set_yscale('log')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('Generalization Comparison: PWL FFNN vs Gaussian Process')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig('gp_vs_pwl_comparison.png')
    print("Comparison plot saved as gp_vs_pwl_comparison.png")
    
    # Save results to CSV for LaTeX
    df_results = pd.DataFrame({
        'Test': labels,
        'PWL_FFNN': pwl_m2,
        'GP': gp_results
    })
    df_results.to_csv('gp_analysis_results.csv', index=False)
