import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_32_neurons():
    bases = ['rand', 'fft', 'haar']
    for base in bases:
        path = f'curves_{base}.txt'
        if not os.path.exists(path): continue
        df = pd.read_csv(path, sep=' ', header=None)
        x = df[0].values
        
        plt.figure(figsize=(10, 6))
        # Plot first 10 neurons for clarity
        for i in range(1, min(11, df.shape[1])):
            plt.plot(x, df[i].values, alpha=0.7, label=f'Neuron {i}')
        
        plt.title(f'Learned Alias Activations (Basis: {base.upper()})')
        plt.xlabel('Input (Pre-activation)')
        plt.ylabel('Output')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'alias_activation_{base}.png')
        plt.close()

def plot_sphere_alias():
    # Curves
    path = 'sphere_curves.txt'
    if os.path.exists(path):
        df = pd.read_csv(path, sep=' ', header=None)
        x = df[0].values
        for i in range(1, df.shape[1]):
            plt.figure(figsize=(10, 6))
            plt.plot(x, df[i].values, color='red' if i==1 else 'blue')
            plt.title(f'Sphere Approximation Alias Curve - Neuron {i}')
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.grid(True)
            plt.savefig(f'sphere_curve_neuron_{i}.png')
            plt.close()

    # Map
    path = 'sphere_map.txt'
    if os.path.exists(path):
        data = np.loadtxt(path)
        plt.figure(figsize=(8, 8))
        plt.imshow(data, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
        plt.colorbar(label='Network Output')
        plt.title('20D Sphere Approximation (2D Slice)')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig('sphere_activation_map.png')
        plt.close()

if __name__ == "__main__":
    plot_32_neurons()
    plot_sphere_alias()
