import pandas as pd
import numpy as np
from scipy.optimize import linprog
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
    mask = ~y.isna()
    X_df = X_df[mask]
    y = y[mask]
    X = (X_df - X_df.mean()) / X_df.std()
    X = X.dropna(axis=1)
    return X, y

def solve_lp(X, y, timeout=30):
    # Subsample 1/10 for speed on 2y data
    X_sub = X.values[::10]
    y_sub = np.sign(y.values[::10])
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
        if capital <= 0:
            return 0.0, 0
    return capital, trades

def optimize_coin(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    X, y = prepare_indicators(df)
    # Train on 80%, test on 20%
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    current_threshold = 0.5
    best_cap = 0.0
    best_round = 0
    
    for r in range(8):
        weights, bias = solve_lp(X_train, y_train)
        if weights is not None:
            cap, trades = simulate(X_test, y_test, weights, bias, current_threshold)
            if cap > best_cap:
                best_cap = cap
                best_round = r + 1
            if cap < 100: current_threshold *= 1.2
            else: current_threshold *= 0.9
        else: break
    return best_cap, best_round

def main():
    data_dir = 'data/ecoins_2y'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    results = []
    for f in files:
        symbol = f.split('_')[0]
        print(f"Optimizing {symbol}...")
        best_cap, br = optimize_coin(os.path.join(data_dir, f))
        print(f"  Best Capital: ${best_cap:.2f} (Round {br})")
        results.append((symbol, best_cap, br))
        
    res_df = pd.DataFrame(results, columns=['Symbol', 'Max_Capital', 'Best_Round'])
    res_df.to_csv('ecoins_2y_summary.txt', sep=' ', index=False)

if __name__ == "__main__":
    main()
