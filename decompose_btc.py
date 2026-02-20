import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

def load_data():
    # BTC data (Naive, assumed UTC from Binance)
    btc = pd.read_csv('data/btc_experiment/btc_1m_3mo.csv', index_col=0, parse_dates=True)
    if btc.index.tz is None:
        btc.index = btc.index.tz_localize('UTC')
    
    # Resample BTC to 10m
    btc_10m = btc.resample('10min').last().ffill()
    btc_returns = btc_10m.pct_change().dropna()
    
    # Futures data (Has +00:00)
    futures = pd.read_csv('data/futures_16_changes.csv', index_col=0, parse_dates=True)
    if futures.index.tz is None:
        futures.index = futures.index.tz_localize('UTC')
    else:
        futures.index = futures.index.tz_convert('UTC')
    
    # Align
    common_idx = btc_returns.index.intersection(futures.index)
    btc_target = btc_returns.loc[common_idx]
    futures_features = futures.loc[common_idx]
    
    return btc_target, futures_features

def iterative_decomposition(y, X_orig):
    y_res = y.values.flatten()
    X = X_orig.values
    symbols = list(X_orig.columns)
    
    history = []
    current_y = y_res.copy()
    
    # We'll use up to 16 steps
    n_steps = min(len(symbols), 16)
    
    for i in range(n_steps):
        best_r2 = -1
        best_idx = -1
        best_pred = None
        
        for j in range(X.shape[1]):
            feat = X[:, j].reshape(-1, 1)
            model = LinearRegression().fit(feat, current_y)
            r2 = model.score(feat, current_y)
            if r2 > best_r2:
                best_r2 = r2
                best_idx = j
                best_pred = model.predict(feat)
        
        symbol = symbols.pop(best_idx)
        history.append({
            "step": i + 1,
            "symbol": symbol,
            "r2_explained": best_r2,
            "residual_var": np.var(current_y - best_pred)
        })
        
        current_y = current_y - best_pred
        
        # Orthogonalize remaining
        chosen_feat = X[:, best_idx].reshape(-1, 1)
        new_X = []
        for j in range(X.shape[1]):
            if j == best_idx: continue
            feat = X[:, j].reshape(-1, 1)
            # Project feat onto chosen_feat and take residual
            proj = LinearRegression().fit(chosen_feat, feat).predict(chosen_feat)
            new_X.append(feat - proj)
        
        if not new_X: break
        X = np.hstack(new_X)
        
        print(f"Step {i+1}: Used {symbol}, R2: {best_r2:.6f}")
        
    return history

def main():
    y, X = load_data()
    print(f"Analyzing {len(y)} aligned 10m intervals...")
    if len(y) == 0:
        print("Error: No overlapping data found. Check date ranges.")
        # Print ranges for debug
        return

    history = iterative_decomposition(y, X)
    
    df_res = pd.DataFrame(history)
    df_res.to_csv('btc_decomposition_results.csv', index=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(df_res['symbol'], df_res['r2_explained'])
    plt.title('BTC Signal Decomposition: Marginal R^2 per Future')
    plt.ylabel('Variance Explained (R^2)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('btc_decomposition_plot.png')

if __name__ == "__main__":
    main()
