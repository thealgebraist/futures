import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
import sys

def run_gp(csv_path, symbol):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # Ensure Close is numeric
    prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values
    
    if len(prices) < 100:
        print(f"{symbol} Not enough data.")
        return

    # Use log returns
    rets = np.log(prices[1:] / prices[:-1])
    
    INPUT_DIM = 64
    X, y = [], []
    for i in range(len(rets) - INPUT_DIM):
        X.append(rets[i : i + INPUT_DIM])
        y.append(rets[i + INPUT_DIM])
    
    X = np.array(X)
    y = np.array(y)
    
    # Subsample for GP speed
    X_train = X[::20]
    y_train = y[::20]
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
    
    gp.fit(X_train, y_train)
    
    last_price = prices[-1]
    current_seq = list(rets[-INPUT_DIM:])
    
    results = [symbol]
    for year in range(1, 11):
        total_log_ret = 0
        for d in range(252):
            pred_ret = gp.predict(np.array(current_seq).reshape(1, -1), return_std=False)[0]
            total_log_ret += pred_ret
            current_seq.pop(0)
            current_seq.append(pred_ret)
        last_price *= np.exp(total_log_ret)
        results.append(str(last_price))
        
    print(" ".join(results))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    run_gp(sys.argv[1], sys.argv[2])
