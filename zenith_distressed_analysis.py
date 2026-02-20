import pandas as pd
import numpy as np
import os

# Robust Analysis of Distressed Ecoins for Recovery Potential

def calculate_recovery_potential(ticker):
    data_dir = '/Users/anders/projects/futures/data/audit'
    csv_path = f"{data_dir}/{ticker}/10m.csv"
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    if len(df) < 1000:
        return None
        
    if 'close' not in df.columns:
        return None
    
    # Exclude delisted/inactive assets (constant price in tail)
    tail_1000 = df.tail(1000)
    if tail_1000['close'].std() < 1e-9:
        return None
        
    df['returns'] = df['close'].pct_change()
    
    # Bottoming Signal: Std Dev of returns in last 1000 candles
    stability = 1.0 / (df['returns'].tail(1000).std() + 1e-8)
    
    # Momentum: Return over last 1000 candles
    momentum = (df['close'].iloc[-1] / df['close'].iloc[-1000]) - 1.0
    
    # Volume Accumulation
    if 'volume' in df.columns and df['volume'].tail(1000).sum() > 1.0:
        vol_ratio = df['volume'].tail(1000).mean() / (df['volume'].mean() + 1e-8)
    else:
        # Fallback metric: Relative Volatility (expansion vs history)
        recent_vol = df['returns'].tail(1000).std()
        overall_vol = df['returns'].std()
        vol_ratio = recent_vol / (overall_vol + 1e-8)
    
    return {
        'ticker': ticker,
        'stability': stability,
        'momentum': momentum,
        'vol_ratio': vol_ratio
    }

# Load distressed list
dist_df = pd.read_csv('/Users/anders/projects/futures/distressed_32_ecoins.csv')
ecoins = dist_df['ticker'].tolist()

# Load alpha from previous audit
alpha_map = {}
if os.path.exists('/Users/anders/projects/futures/zenith_ecosystem_results.csv'):
    alpha_df = pd.read_csv('/Users/anders/projects/futures/zenith_ecosystem_results.csv')
    alpha_map = dict(zip(alpha_df['ticker'], alpha_df['alpha']))

results = []
for ecoin in ecoins:
    metrics = calculate_recovery_potential(ecoin)
    if metrics:
        metrics['alpha'] = alpha_map.get(ecoin, 1.0)
        # Recovery Score
        metrics['recovery_score'] = (
            metrics['stability'] * 0.2 + 
            metrics['momentum'] * 100.0 + 
            metrics['vol_ratio'] * 0.5 + 
            (metrics['alpha'] - 1.0) * 10.0
        )
        results.append(metrics)

final_df = pd.DataFrame(results).sort_values(by='recovery_score', ascending=False)
print("Top Recovery Candidates (Distressed Ecoins - CLEANED V2):")
print(final_df[['ticker', 'recovery_score', 'momentum', 'vol_ratio']].head(16))
final_df.to_csv('/Users/anders/projects/futures/distressed_recovery_analysis.csv', index=False)
