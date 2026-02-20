import pandas as pd
import numpy as np
import os

def analyze():
    # Load 2y daily
    df = pd.read_csv('data/benchmarks/indices_2y_daily.csv', index_col=0, parse_dates=True)
    
    # CAGR and Sharpe
    results = []
    risk_free_rate = 0.04 # 4% proxy
    
    for col in df.columns:
        prices = df[col].dropna()
        if len(prices) < 2: continue
        
        # Total Return
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        # Years
        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        # CAGR
        cagr = (1 + total_return)**(1/years) - 1
        
        # Daily Returns
        daily_rets = prices.pct_change().dropna()
        ann_vol = daily_rets.std() * np.sqrt(252)
        sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        # Test Window Performance (Last 20% of 60 days ~ 12 days)
        # We'll use the daily data for this proxy comparison
        test_window_start = int(len(prices) * 0.9) # Approx last 2 weeks
        test_return = (prices.iloc[-1] / prices.iloc[test_window_start]) - 1
        
        results.append({
            'Symbol': col,
            'CAGR': cagr,
            'Vol': ann_vol,
            'Sharpe': sharpe,
            'Test_Window_Ret': test_return
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('benchmark_analysis_results.csv', index=False)
    print(res_df.to_string())

if __name__ == "__main__":
    analyze()
