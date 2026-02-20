import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_curves():
    bases = ['rand', 'fft', 'haar']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for idx, base in enumerate(bases):
        path = f'curves_{base}.txt'
        df = pd.read_csv(path, sep=' ', header=None)
        x = df[0].values
        # Plot a subset of neurons for clarity
        for i in range(1, min(11, df.shape[1])):
            axes[idx].plot(x, df[i].values, alpha=0.7, label=f'Neuron {i}')
        
        axes[idx].set_title(f'Learned Alias Activations (Basis: {base.upper()})')
        axes[idx].set_xlabel('Input (Pre-activation)')
        axes[idx].set_ylabel('Output')
        axes[idx].grid(True)
        # axes[idx].legend()
        
    plt.tight_layout()
    plt.savefig('learned_alias_activations.png')

if __name__ == "__main__":
    plot_curves()
