import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_eigen():
    df = pd.read_csv('data/india_etf/smin_15m.csv', index_col=0, parse_dates=True)
    # Cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    returns = df['Log_Return'].values
    
    lags = 10
    n = len(returns)
    if n < lags + 100:
        print("Not enough data for analysis.")
        return

    X = np.zeros((n - lags, lags))
    for i in range(lags):
        X[:, i] = returns[lags-i-1 : n-i-1]
        
    window_size = 100
    eigenvalues = []
    
    for i in range(len(X) - window_size):
        window = X[i : i + window_size]
        # Check for constant columns (zero std) which break corrcoef
        if np.any(np.std(window, axis=0) < 1e-9):
            # Use Covariance instead if variance is too low for correlation
            cov_mat = np.cov(window, rowvar=False)
            eig_vals = np.linalg.eigvalsh(cov_mat)
        else:
            try:
                corr_mat = np.corrcoef(window, rowvar=False)
                eig_vals = np.linalg.eigvalsh(corr_mat)
            except np.linalg.LinAlgError:
                # Fallback to zeros if convergence fails
                eig_vals = np.zeros(lags)
                
        eig_vals = np.sort(eig_vals)[::-1]
        eigenvalues.append(eig_vals)
        
    eigenvalues = np.array(eigenvalues)
    
    plt.figure(figsize=(12, 6))
    if len(eigenvalues) > 0:
        plt.plot(eigenvalues[:, 0], label='Max Eigenvalue')
        plt.plot(eigenvalues[:, 1], label='2nd Eigenvalue')
        plt.plot(eigenvalues[:, -1], label='Min Eigenvalue')
    plt.title('Rolling Eigenvalue Spectrum of SMIN (Robust)')
    plt.savefig('data/india_etf/smin_eigen_spectrum.png')
    
    with open("smin_analysis_results.txt", "w") as f:
        if len(eigenvalues) > 0:
            f.write(f"Mean_Max_Eigenvalue: {np.mean(eigenvalues[:, 0]):.4f}\n")
            f.write(f"Total_Windows: {len(eigenvalues)}\n")
        else:
            f.write("Analysis_Failed: True\n")

if __name__ == "__main__":
    analyze_eigen()
