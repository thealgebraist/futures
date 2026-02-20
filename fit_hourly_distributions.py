import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def fit_and_plot_hourly_distributions():
    print("Loading hourly returns data...")
    hourly_df = pd.read_csv('data/hourly_returns.csv')
    
    # Remove extreme outliers as before
    hourly_df = hourly_df[np.abs(hourly_df['return_value']) < 0.1]

    # Store results
    hourly_params = []

    # Plot specific hours as examples
    example_hours = [2, 14] # 02:00 and 14:00 as requested implicitly
    
    for hour in range(24):
        print(f"\nProcessing hour {hour:02d}:00...")
        hour_returns = hourly_df[hourly_df['hour_of_day'] == hour]['return_value'].values
        
        if len(hour_returns) < 100:
            print(f"  Not enough samples for hour {hour:02d}:00. Skipping.")
            continue
            
        print(f"  Total samples for hour {hour:02d}:00: {len(hour_returns)}")
        
        # Fit Gaussian
        mu_norm, std_norm = stats.norm.fit(hour_returns)
        
        # Fit Cauchy
        loc_cauchy, scale_cauchy = stats.cauchy.fit(hour_returns)
        
        # Fit Student-t
        df_t, loc_t, scale_t = stats.t.fit(hour_returns)
        
        # Store parameters
        hourly_params.append({
            'hour': hour,
            'gaussian_mu': mu_norm,
            'gaussian_std': std_norm,
            'cauchy_loc': loc_cauchy,
            'cauchy_scale': scale_cauchy,
            'student_t_df': df_t,
            'student_t_loc': loc_t,
            'student_t_scale': scale_t
        })
        
        print(f"  Gaussian: mu={mu_norm:.6f}, std={std_norm:.6f}")
        print(f"  Cauchy: loc={loc_cauchy:.6f}, scale={scale_cauchy:.6f}")
        print(f"  Student-t: df={df_t:.2f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
        
        # Plot for example hours
        if hour in example_hours:
            plt.figure(figsize=(10, 6))
            count, bins, ignored = plt.hist(hour_returns, bins=200, density=True, alpha=0.5, color='gray', label='Empirical')
            x = np.linspace(min(bins), max(bins), 1000)
            
            plt.plot(x, stats.norm.pdf(x, mu_norm, std_norm), 'r-', lw=2, label=f'Gaussian (std={std_norm:.4f})')
            plt.plot(x, stats.cauchy.pdf(x, loc_cauchy, scale_cauchy), 'b--', lw=2, label=f'Cauchy (scale={scale_cauchy:.4f})')
            plt.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'g-', lw=2, label=f'Student-t (df={df_t:.2f})')
            
            plt.title(f'Distribution of 10m Log-Returns for Hour {hour:02d}:00')
            plt.xlabel('Log-Return')
            plt.ylabel('Density (log scale)')
            plt.legend()
            plt.yscale('log')
            plt.ylim(max(1e-6, plt.ylim()[0]), plt.ylim()[1])
            plt.grid(True, which="both", ls="-", alpha=0.5)
            
            output_png = f'hourly_distribution_h{hour:02d}.png'
            plt.savefig(output_png)
            plt.close()
            print(f"  Saved plot to {output_png}")
            
    hourly_params_df = pd.DataFrame(hourly_params)
    os.makedirs('data', exist_ok=True)
    hourly_params_df.to_csv('data/hourly_distribution_params.csv', index=False)
    print("\nSaved hourly distribution parameters to data/hourly_distribution_params.csv")

if __name__ == "__main__":
    fit_and_plot_hourly_distributions()
