import pandas as pd
import numpy as np
import os

# 1. Simulate Data
np.random.seed(42) # for reproducibility
days = 252 # ~1 year of trading days

# Simulate AAPL price (starts at 170, drift 0.05% per day, std dev 1%)
aapl_returns = np.random.normal(0.0005, 0.01, days)
aapl_prices = 170 * np.exp(np.cumsum(aapl_returns))

# Simulate TSM price (positively correlated with AAPL, starts at 100, drift 0.04% per day, std dev 1.2%)
# Introduce some correlation by adding a portion of AAPL's returns
tsm_base_returns = np.random.normal(0.0004, 0.012, days)
tsm_returns = 0.6 * aapl_returns + 0.4 * tsm_base_returns # 60% correlation with AAPL
tsm_prices = 100 * np.exp(np.cumsum(tsm_returns))

# Simulate USD/CNY exchange rate (starts at 7.0, slight negative drift for USD, std dev 0.3%)
# Manufacturing costs in CNY mean a stronger CNY (lower USD/CNY) is bad for US-based AAPL
usdcny_returns = np.random.normal(-0.0001, 0.003, days)
usdcny_rates = 7.0 * np.exp(np.cumsum(usdcny_returns))

# Create DataFrame
dates = pd.date_range(start='2025-01-01', periods=days, freq='B') # Business days
df = pd.DataFrame({
    'AAPL_Price': aapl_prices,
    'TSM_Price': tsm_prices,
    'USDCNY_Rate': usdcny_rates
}, index=dates)

# 2. Calculate Log Returns
df['AAPL_Return'] = np.log(df['AAPL_Price'] / df['AAPL_Price'].shift(1))
df['TSM_Return'] = np.log(df['TSM_Price'] / df['TSM_Price'].shift(1))
df['USDCNY_Change'] = np.log(df['USDCNY_Rate'] / df['USDCNY_Rate'].shift(1))

# Drop first row with NaN returns
df = df.dropna()

# 3. Analyze Correlations
aapl_vs_tsm_corr = df['AAPL_Return'].corr(df['TSM_Return'])
aapl_vs_usdcny_corr = df['AAPL_Return'].corr(df['USDCNY_Change'])

# 4. Summarize Results
print("--- Simulated Dependency Analysis for Apple (AAPL) ---")
print(f"Period: 1 year of daily data ({days} trading days)")
print("""
Dependencies considered:
  - Key Supplier: TSMC (TSM) - for chip manufacturing
  - Foreign Exchange: USD/CNY - impact on manufacturing costs in China
""")
print("Correlation Results (Pearson):")
print(f"  - AAPL Daily Returns vs. TSM Daily Returns: {aapl_vs_tsm_corr:.4f}")
print(f"  - AAPL Daily Returns vs. USD/CNY Daily Change: {aapl_vs_usdcny_corr:.4f}")
print(f"""
Interpretation of Simulated Results:
  - A positive correlation with TSM ({aapl_vs_tsm_corr:.4f}) suggests that strong performance in Apple's primary chip supplier generally aligns with Apple's stock performance.
  - A negative correlation with USD/CNY ({aapl_vs_usdcny_corr:.4f}) would imply that when the USD strengthens against the CNY (USDCNY rate goes up), Apple's stock tends to perform worse (or vice versa). This could be due to increased manufacturing costs in USD terms for components sourced in CNY, or decreased purchasing power for Chinese consumers.

Note: These results are based on simulated data and serve as a conceptual demonstration.""")
