import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os

def load_and_prepare():
    df = pd.read_csv('data/sol_experiment/sol_1m_3mo.csv', index_col=0, parse_dates=True)
    df = df.sort_index()
    
    # Target: Sign of next 10-min return
    df['Next_Ret_10m'] = df['Close'].shift(-10) / df['Close'] - 1
    df = df.dropna()
    
    # Features: Returns at various lags
    lags = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]
    features = {}
    for lag in lags:
        features[f'Ret_{lag}m'] = df['Close'] / df['Close'].shift(lag) - 1
    
    # Volatility proxy
    features['Vol_10m'] = df['Close'].pct_change().rolling(10).std()
    features['Vol_60m'] = df['Close'].pct_change().rolling(60).std()
    
    X_df = pd.DataFrame(features)
    X_df = X_df.dropna()
    
    common_idx = X_df.index.intersection(df.index)
    X = X_df.loc[common_idx]
    y = df.loc[common_idx, 'Next_Ret_10m']
    
    X = (X - X.mean()) / X.std()
    return X, y

def solve_lp(X, y):
    X_sub = X.iloc[::10].values
    y_sub = np.sign(y.iloc[::10].values)
    n_samples, n_features = X_sub.shape
    
    c = np.concatenate([np.zeros(n_features + 1), np.ones(n_samples)])
    A_ub = np.zeros((n_samples, n_features + 1 + n_samples))
    for t in range(n_samples):
        A_ub[t, :n_features] = -y_sub[t] * X_sub[t]
        A_ub[t, n_features] = -y_sub[t] 
        A_ub[t, n_features + 1 + t] = -1.0 
        
    b_ub = -np.ones(n_samples)
    bounds = [(None, None)] * (n_features + 1) + [(0, None)] * n_samples
    
    print(f"Solving LP with {n_samples} samples...")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[:n_features], res.x[n_features]
    else:
        print(f"LP Optimization failed: {res.message}")
        return None, None

def main():
    X, y = load_and_prepare()
    print(f"Prepared {len(X)} samples.")
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    weights, bias = solve_lp(X_train, y_train)
    if weights is not None:
        indicator_names = X.columns
        results = pd.DataFrame({'Indicator': indicator_names, 'Weight': weights})
        print("\nOptimal Indicator Weights:")
        print(results.to_string())
        
        test_signals = np.dot(X_test.values, weights) + bias
        capital = 100.0
        thresh = 0.5 
        history = [capital]
        for i in range(len(test_signals)):
            if i % 10 == 0:
                signal = test_signals[i]
                ret = y_test.iloc[i]
                if signal > thresh:
                    capital += capital * ret - 0.1 
                elif signal < -thresh:
                    capital -= capital * ret + 0.1 
                history.append(capital)
                if capital <= 0:
                    capital = 0
                    break
        print(f"\nFinal Capital: ${capital:.2f}")
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.title('SOL LP Trading Simulation')
        plt.savefig('sol_lp_simulation.png')

if __name__ == "__main__":
    main()
