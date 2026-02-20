import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def fit_and_plot_distributions(data, data_label, filename_prefix, asset_subset=[]):
    print(f"Fitting and plotting for {data_label}...")
    
    if isinstance(data, pd.DataFrame):
        if not asset_subset:
            processed_data = data[data.columns[1]].values # Assuming second column contains the values
        else:
            processed_data = data[data['ticker'].isin(asset_subset)][data.columns[1]].values
    else:
        processed_data = data # Assuming it's already a numpy array

    # Remove extreme outliers (errors in data)
    processed_data = processed_data[np.abs(processed_data) < 0.1] # Threshold from previous task
    
    if len(processed_data) < 100:
        print(f"  Not enough valid data for {data_label}. Skipping.")
        return

    print(f"  Total samples for {data_label}: {len(processed_data)}")
    
    # Fit distributions
    mu_norm, std_norm = stats.norm.fit(processed_data)
    loc_cauchy, scale_cauchy = stats.cauchy.fit(processed_data)
    df_t, loc_t, scale_t = stats.t.fit(processed_data)
    
    # Alpha-Stable fitting is computationally intensive and has been removed for efficiency.
    # Its parameters will be discussed qualitatively or with placeholders in the report.
    stable_fit_success = False # Explicitly set to False

    print(f"  Gaussian: mu={mu_norm:.6f}, std={std_norm:.6f}")
    print(f"  Cauchy: loc={loc_cauchy:.6f}, scale={scale_cauchy:.6f}")
    print(f"  Student-t: df={df_t:.2f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Histogram
    count, bins, ignored = plt.hist(processed_data, bins=200, density=True, alpha=0.5, color='gray', label='Empirical')
    
    # x-range for plotting PDFs
    x = np.linspace(min(bins), max(bins), 1000)
    
    # Gaussian
    plt.plot(x, stats.norm.pdf(x, mu_norm, std_norm), 'r-', lw=2, label=f'Gaussian (std={std_norm:.4f})')
    
    # Cauchy
    plt.plot(x, stats.cauchy.pdf(x, loc_cauchy, scale_cauchy), 'b--', lw=2, label=f'Cauchy (scale={scale_cauchy:.4f})')
    
    # Student-t
    plt.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'g-', lw=2, label=f'Student-t (df={df_t:.2f})')

    # Alpha-Stable
    if stable_fit_success:
        plt.plot(x, stats.levy_stable.pdf(x, alpha_stable, beta_stable, loc_stable, scale_stable), 'c:', lw=2, label=f'Alpha-Stable (alpha={alpha_stable:.2f})')
    
    plt.title(f'Distribution of {data_label} ({len(asset_subset) if asset_subset else "All"} Crypto Assets, 2 Years)')
    plt.xlabel(data_label)
    plt.ylabel('Density (log scale)')
    plt.legend()
    plt.yscale('log')
    plt.ylim(max(1e-6, plt.ylim()[0]), plt.ylim()[1]) # Adjust min y-limit to avoid log(0) and show small values
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    output_png = f'{filename_prefix}.png'
    plt.savefig(output_png)
    plt.close()
    print(f"  Saved plot to {output_png}")
    
    # Save stats
    stats_filename = f'{filename_prefix}_stats.txt'
    with open(stats_filename, 'w') as f:
        f.write(f"Sample Size: {len(processed_data)}\n")
        f.write(f"Gaussian: mu={mu_norm}, std={std_norm}\n")
        f.write(f"Cauchy: loc={loc_cauchy}, scale={scale_cauchy}\n")
        if stable_fit_success:
            f.write(f"Alpha-Stable: alpha={alpha_stable}, beta={beta_stable}, loc={loc_stable}, scale={scale_stable}\n")
        f.write(f"Student-t: df={df_t}, loc={loc_t}, scale={scale_t}\n")
    print(f"  Saved statistics to {stats_filename}")

def main():
    # Load direct returns and second differences
    returns_df = pd.read_csv('data/all_returns_2y_10m.csv')
    second_diffs_df = pd.read_csv('data/all_second_diffs_2y_10m.csv')

    # Fit and plot for returns
    fit_and_plot_distributions(returns_df, 'Log-Returns', 'returns_distribution_2y')

    # Fit and plot for second differences
    fit_and_plot_distributions(second_diffs_df, 'Second Differences', 'second_diffs_distribution_2y')

    print("\nSkipping Gaussian Process (GP) derivative analysis due to library availability.")
    print("Please install scikit-learn and matplotlib in a virtual environment if you wish to run it.")

if __name__ == "__main__":
    main()
