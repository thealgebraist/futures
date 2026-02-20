import pandas as pd
import numpy as np
import os

def prepare_data():
    df = pd.read_csv('data/futures_10m_v2.csv', index_col=0)
    # Target micro contracts or equivalent top 4
    # We'll use ES, NQ, CL, GC as proxies for their micro versions
    data = df.iloc[:, [0, 1, 3, 4]].values
    
    # Normalization
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / std
    
    input_steps = 16
    output_steps = 4
    
    # 60 days total. Let's use first 30 days as base training.
    # The remaining 30 days are for 8-iteration walk-forward.
    total_samples = len(data_norm)
    base_split = total_samples // 2
    test_samples = total_samples - base_split
    iter_size = test_samples // 8
    
    os.makedirs('data/walkforward', exist_ok=True)
    
    for i in range(8):
        train_end = base_split + i * iter_size
        test_end = train_end + iter_size
        
        # Training segment
        train_data = data_norm[:train_end]
        # Test segment
        test_data = data_norm[train_end - input_steps : test_end]
        
        def segment(d):
            X, y = [], []
            for j in range(len(d) - input_steps - output_steps + 1):
                X.append(d[j : j + input_steps].flatten())
                y.append(d[j + input_steps : j + input_steps + output_steps].flatten())
            return np.array(X), np.array(y)
            
        X_train, y_train = segment(train_data)
        X_test, y_test = segment(test_data)
        
        # Save as npy
        np.save(f'data/walkforward/X_train_{i}.npy', X_train.astype(np.float32))
        np.save(f'data/walkforward/y_train_{i}.npy', y_train.astype(np.float32))
        np.save(f'data/walkforward/X_test_{i}.npy', X_test.astype(np.float32))
        np.save(f'data/walkforward/y_test_{i}.npy', y_test.astype(np.float32))
        
    print(f"Prepared 8 walk-forward iterations. Iteration size: {iter_size} samples (~3.5 days).")

if __name__ == "__main__":
    prepare_data()
