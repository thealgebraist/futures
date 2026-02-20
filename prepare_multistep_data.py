import pandas as pd
import numpy as np
import os

def prepare_data():
    if not os.path.exists('data/futures_10m_v2.csv'):
        print("Source data not found.")
        return
        
    df = pd.read_csv('data/futures_10m_v2.csv', index_col=0)
    # ES, NQ, CL, GC are at indices 0, 1, 3, 4
    top_4 = df.iloc[:, [0, 1, 3, 4]].values
    
    # Normalize
    mean = top_4.mean(axis=0)
    std = top_4.std(axis=0)
    top_4_norm = (top_4 - mean) / std
    
    input_steps = 16
    output_steps = 4
    n_features = 4
    
    X, y = [], []
    for i in range(len(top_4_norm) - input_steps - output_steps + 1):
        X.append(top_4_norm[i : i + input_steps].flatten())
        y.append(top_4_norm[i + input_steps : i + input_steps + output_steps].flatten())
        
    X = np.array(X)
    y = np.array(y)
    
    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    os.makedirs('data/multistep', exist_ok=True)
    np.save('data/multistep/X_train.npy', X_train.astype(np.float32))
    np.save('data/multistep/y_train.npy', y_train.astype(np.float32))
    np.save('data/multistep/X_test.npy', X_test.astype(np.float32))
    np.save('data/multistep/y_test.npy', y_test.astype(np.float32))
    
    print(f"Prepared multistep data: {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

if __name__ == "__main__":
    prepare_data()
