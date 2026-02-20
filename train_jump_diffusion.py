import numpy as np
import pandas as pd
import time

def merton_jump_diffusion(S0, mu, sigma, lamb, m, v, T, steps, paths):
    dt = T / steps
    # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z + sum(Y_i))
    # where Y_i ~ N(m, v)
    
    paths_matrix = np.zeros((paths, steps + 1))
    paths_matrix[:, 0] = S0
    
    for i in range(steps):
        Z = np.random.standard_normal(paths)
        N = np.random.poisson(lamb * dt, paths)
        
        # Log-jumps: sum of N independent Gaussians
        # Each jump Y_i ~ N(m, v) -> sum(Y_i) ~ N(N*m, N*v)
        jump_sum = np.zeros(paths)
        for p in range(paths):
            if N[p] > 0:
                jump_sum[p] = np.random.normal(N[p] * m, np.sqrt(N[p] * v))
        
        paths_matrix[:, i+1] = paths_matrix[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + jump_sum
        )
        
    return paths_matrix

def main():
    # Load data to estimate parameters
    df = pd.read_csv('data/futures_10m_v2.csv', index_col=0)
    returns = np.log(df.iloc[:, 0] / df.iloc[:, 0].shift(1)).dropna()
    
    # Simple estimation (naive)
    mu = returns.mean() * 252 * 24 * 6 # annualized
    sigma = returns.std() * np.sqrt(252 * 24 * 6)
    
    print("Merton Jump Diffusion MC Simulation (60s)...")
    start = time.time()
    
    # 60s of simulations
    count = 0
    while time.time() - start < 60:
        # Simulate 1000 paths for the next 4 steps
        paths = merton_jump_diffusion(S0=df.iloc[-1, 0], mu=mu, sigma=sigma, lamb=0.1, m=0, v=0.01, T=4/144, steps=4, paths=1000)
        count += 1
        
    print(f"Executed {count} MC simulations batches in 60s.")
    
    with open("mc_results.txt", "w") as f:
        f.write(f"MC_Batches: {count}\n")
        f.write(f"Final_Price_Mean: {paths[:, -1].mean()}\n")

if __name__ == "__main__":
    main()
