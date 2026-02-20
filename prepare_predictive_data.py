import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def prepare_data():
    # Load data
    file_path = 'data/futures_1m_4mo.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    target_col = 'NQ=F'
    values = df[target_col].values
    
    # Remove NaNs
    values = values[~np.isnan(values)]
    
    # Normalize
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    
    # Create windows
    input_steps = 16
    output_steps = 4
    
    X = []
    y = []
    
    for i in range(len(values_scaled) - input_steps - output_steps + 1):
        X.append(values_scaled[i : i + input_steps])
        y.append(values_scaled[i + input_steps : i + input_steps + output_steps])
        
    X = np.array(X)
    y = np.array(y)
    
    # Split 80/20
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train_p.npy', X_train)
    np.save('data/X_test_p.npy', X_test)
    np.save('data/y_train_p.npy', y_train)
    np.save('data/y_test_p.npy', y_test)
    
    print(f"Data prepared from {file_path}:")
    print(f"{len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")

if __name__ == "__main__":
    prepare_data()
