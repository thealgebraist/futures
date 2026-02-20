import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os

def prepare_indicators(df):
    lags = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440]
    features = {}
    for lag in lags:
        features[f'Ret_{lag}m'] = df['Close'] / df['Close'].shift(lag//10) - 1
    for window in [10, 20, 50, 100]:
        ma = df['Close'].rolling(window).mean()
        features[f'MA_{window}_Dev'] = df['Close'] / ma - 1
    features['Vol_100'] = df['Close'].pct_change().rolling(100).std()
    X_df = pd.DataFrame(features)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df['Close'].shift(-1) / df['Close'] - 1
    y = y.loc[X_df.index]
    # Filter out target NaNs if any
    mask = ~y.isna()
    X_df = X_df[mask]
    y = y[mask]
    # Normalize features, drop zero-std
    X = (X_df - X_df.mean()) / X_df.std()
    X = X.dropna(axis=1) # Drop columns that became NaN (zero std)
    return X, y

def solve_lp(X, y, timeout=60):
    X_sub = X.values[::5]
    y_sub = np.sign(y.values[::5])
    # Filter y_sub == 0
    mask = y_sub != 0
    X_sub = X_sub[mask]
    y_sub = y_sub[mask]
    
    n_samples, n_features = X_sub.shape
    c = np.concatenate([np.zeros(n_features + 1), np.ones(n_samples)])
    A_ub = np.zeros((n_samples, n_features + 1 + n_samples))
    for t in range(n_samples):
        A_ub[t, :n_features] = -y_sub[t] * X_sub[t]
        A_ub[t, n_features] = -y_sub[t] 
        A_ub[t, n_features + 1 + t] = -1.0 
    b_ub = -np.ones(n_samples)
    bounds = [(None, None)] * (n_features + 1) + [(0, None)] * n_samples
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'time_limit': timeout})
    if res.success or res.status == 1:
        return res.x[:n_features], res.x[n_features]
    return None, None

def simulate(X_test, y_test, weights, bias, threshold):
    signals = np.dot(X_test.values, weights) + bias
    capital = 100.0
    history = [capital]
    trades = 0
    for i in range(len(signals)):
        sig = signals[i]
        ret = y_test.iloc[i]
        if sig > threshold:
            capital += capital * ret - 0.05
            trades += 1
        elif sig < -threshold:
            capital -= capital * ret + 0.05
            trades += 1
        if i % 10 == 0: history.append(capital)
        if capital <= 0:
            capital = 0
            break
    return capital, history, trades

def main():
    df = pd.read_csv('data/sol_experiment/sol_10m_1y.csv', index_col=0, parse_dates=True)
    df = df.sort_index()
    X, y = prepare_indicators(df)
    print(f"Total samples: {len(X)}")
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    current_threshold = 0.5
    results = []
    for r in range(8):
        print(f"\n--- Round {r+1} ---")
        weights, bias = solve_lp(X_train, y_train, timeout=60)
        if weights is not None:
            cap, hist, trades = simulate(X_test, y_test, weights, bias, current_threshold)
            print(f"Final Capital: ${cap:.2f} | Trades: {trades} | Threshold: {current_threshold:.4f}")
            results.append((r+1, cap, trades, current_threshold))
            if cap < 100: current_threshold *= 1.2
            else: current_threshold *= 0.9
        else: print("LP failed.")
    res_df = pd.DataFrame(results, columns=['Round', 'Capital', 'Trades', 'Threshold'])
    res_df.to_csv('sol_1y_lp_results.txt', sep=' ', index=False)
    if not res_df.empty:
        best_round = res_df.loc[res_df['Capital'].idxmax()]
        print("\nBest Result:")
        print(best_round)

if __name__ == "__main__":
    main()
