import pandas as pd
import numpy as np

def analyze():
    try:
        data = pd.read_csv('nonlinear_scaling_results.txt', sep=' ', header=None, names=['problem', 'width', 'mse'])
    except Exception as e:
        print(f"Error reading results: {e}")
        return

    # Normalize MSE per problem (Z-score or Min-Max) to avoid high-MSE problems dominating
    def normalize(group):
        group['norm_mse'] = (group['mse'] - group['mse'].min()) / (group['mse'].max() - group['mse'].min() + 1e-9)
        return group

    data = data.groupby('problem', group_keys=False).apply(normalize)
    
    # Average normalized MSE per width
    summary = data.groupby('width')['norm_mse'].mean().reset_index()
    summary = summary.sort_values('norm_mse')
    
    print("Generalization Sweet Spot Analysis:")
    print(summary.to_string(index=False))
    
    best_width = summary.iloc[0]['width']
    print(f"
Best Width across 32 problems: n={int(best_width)}")

if __name__ == "__main__":
    analyze()
