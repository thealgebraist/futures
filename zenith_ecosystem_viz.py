import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('zenith_ecosystem_results.csv', header=None, names=['ticker', 'roi', 'trades', 'exposure', 'alpha'])

# Convert to numeric, errors='coerce' to handle any malformed data
for col in ['roi', 'trades', 'exposure', 'alpha']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)
df_clean = df[df['alpha'] > 1.0].copy()

plt.figure(figsize=(12, 7))

# Scatter plot: Alpha vs ROI
# Alpha of 2.0 is Gaussian. Lower alpha = fatter tails.
scatter = plt.scatter(df_clean['alpha'], df_clean['roi'], c=df_clean['roi'], cmap='RdYlGn', s=100, edgecolors='black', alpha=0.8)
plt.colorbar(scatter, label='ROI (%)')

# Annotate top performers
top_performers = df_clean.sort_values('roi', ascending=False).head(8)
for i, row in top_performers.iterrows():
    plt.annotate(row['ticker'], (row['alpha'], row['roi']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(2.0, color='blue', linestyle='--', alpha=0.3, label=r'Gaussian ($\alpha=2.0$)')

plt.title("Zenith Ecosystem Audit: Alpha Stability vs Fee-Adjusted ROI", fontsize=14)
plt.xlabel("Lévy Stability Index ($\\alpha$)", fontsize=12)
plt.ylabel("ROI (%) [Net of 5bps Fees]", fontsize=12)
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()

plt.savefig('zenith_ecosystem_scatter.png')
plt.close()

# Generate a top-10 table for LaTeX
print("Top 10 Performers:")
print(df_clean.sort_values('roi', ascending=False).head(10)[['ticker', 'roi', 'alpha', 'trades']].to_latex(index=False))
