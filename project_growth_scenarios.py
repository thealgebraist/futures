import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def project_scenarios():
    # Load analysis results
    df = pd.read_csv('benchmark_analysis_results.csv', index_col=0)
    
    # We will project SPY and QQQ
    symbols = ['SPY', 'QQQ']
    horizon_years = 5
    initial_cap = 100.0
    
    plt.figure(figsize=(10, 6))
    
    for symbol in symbols:
        if symbol not in df.index: continue
        cagr = df.loc[symbol, 'CAGR']
        vol = df.loc[symbol, 'Vol']
        
        # Mean return (geometric to arithmetic approximation)
        mu = cagr + 0.5 * vol**2
        
        # Projections (Optimistic, Mean, Conservative)
        # Assuming Normal distribution of log returns
        time_points = np.linspace(0, horizon_years, 61) # Monthly steps
        
        # Mean projection: e^(mu * t)
        mean_proj = initial_cap * np.exp((mu - 0.5 * vol**2) * time_points)
        
        # Conservative (10th percentile): e^( (mu - 0.5*vol^2)*t - 1.28*vol*sqrt(t) )
        cons_proj = initial_cap * np.exp((mu - 0.5 * vol**2) * time_points - 1.28 * vol * np.sqrt(time_points))
        
        # Optimistic (90th percentile): e^( (mu - 0.5*vol^2)*t + 1.28*vol*sqrt(t) )
        opt_proj = initial_cap * np.exp((mu - 0.5 * vol**2) * time_points + 1.28 * vol * np.sqrt(time_points))
        
        plt.plot(time_points, mean_proj, label=f'{symbol} Mean', linewidth=2)
        plt.fill_between(time_points, cons_proj, opt_proj, alpha=0.1, label=f'{symbol} 10-90% Band')
        
        print(f"{symbol} 5Y Projection: Mean ${mean_proj[-1]:.2f} (Range ${cons_proj[-1]:.2f} - ${opt_proj[-1]:.2f})")
    
    plt.title('5-Year Growth Scenarios (Monte Carlo Proxy)')
    plt.xlabel('Years')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('index_growth_projections.png')

if __name__ == "__main__":
    project_scenarios()
