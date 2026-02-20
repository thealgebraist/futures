import os
import re

def categorize_pdfs(pdf_dir='/Users/anders/projects/pdf'):
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    themes = {
        'Financial/Trading': ['futures', 'trading', 'mc_report', 'predictive', 'short_only', 'btcusdt', 'btc', 'mes', 'en-mini', 'backtest', 'profit', 'equity'],
        'Neural Architectures/Activations': ['activation', 'ffnn', 'pwl', 'haar', 'spectral', 'layer', 'neuron', 'decomposition', 'gabor', 'lstm', 'ode', 'ltc', 'liquid'],
        'Semantics/Formal Methods': ['semantics', 'coq', 'interpreter', 'equivalence', 'grammar', 'gutenberg', 'alice', 'sentence', 'parse', 'axioms'],
        'Physical/Biological Systems': ['dna', '2body', 'gravitational', 'sphere', 'spherical', 'station', 'room', 'canteen'],
        'Optimization/Solvers': ['optimization', 'bench', 'l-bfgs', 'adam', 'gradient', 'solver', 'amx', 'quantized'],
        'Visualization/Game Engine': ['summary', 'camera', 'voxel', 'vector', 'crt', 'rgba', 'zoom', 'sd_distillation'],
        'General/Meta': ['report', 'final', 'summary', 'comprehensive', 'overview']
    }
    
    categorized = {k: [] for k in themes.keys()}
    categorized['Other'] = []
    
    for pdf in pdfs:
        found = False
        lower_name = pdf.lower()
        for theme, keywords in themes.items():
            if any(kw in lower_name for kw in keywords):
                categorized[theme].append(pdf)
                found = True
                break
        if not found:
            categorized['Other'].append(pdf)
            
    for theme, files in categorized.items():
        print(f"### {theme} ({len(files)} files)")
        for f in sorted(files)[:10]: # Print top 10 for overview
            print(f"- {f}")
        if len(files) > 10:
            print(f"- ... and {len(files) - 10} more.")

if __name__ == "__main__":
    categorize_pdfs()
