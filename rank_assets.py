import pandas as pd

df = pd.read_csv('/Users/anders/projects/futures/zenith_ecosystem_results.csv')
top_16 = df.sort_values(by='roi', ascending=False).head(16)
print(top_16[['ticker', 'roi']])
top_16.to_csv('/Users/anders/projects/futures/top_16_roi_assets.csv', index=False)
