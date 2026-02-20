import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def plot_fitting(ticker):
    path = f"data/returns_1y/{ticker}_returns.txt"
    if not os.path.exists(path):
        return
    
    returns = np.loadtxt(path)
    # Remove outliers for better plotting
    q1 = np.percentile(returns, 1)
    q99 = np.percentile(returns, 99)
    filtered = returns[(returns > q1) & (returns < q99)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(filtered, bins=100, density=True, alpha=0.5, label='Actual Returns (98% central)')
    
    x = np.linspace(min(filtered), max(filtered), 1000)
    
    # Gaussian
    mu, std = stats.norm.fit(returns)
    plt.plot(x, stats.norm.pdf(x, mu, std), 'r-', label='Gaussian Fit')
    
    # Cauchy
    loc_c, scale_c = stats.cauchy.fit(returns)
    plt.plot(x, stats.cauchy.pdf(x, loc_c, scale_c), 'g-', label='Cauchy Fit')
    
    # Student-t
    df_t, loc_t, scale_t = stats.t.fit(returns)
    plt.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'b-', label=f'Student-t Fit (df={df_t:.2f})')
    
    plt.title(f"{ticker} Return Distribution (10-min interval)")
    plt.xlabel("Log-Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{ticker}_distribution.png")
    plt.close()

def main():
    plot_fitting('BTCUSDT')
    plot_fitting('SOLUSDT')

if __name__ == "__main__":
    main()
