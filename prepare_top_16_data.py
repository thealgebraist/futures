import pandas as pd
import subprocess
import os

# Top 16 assets from initial ranking
assets = [
    'FLOWUSDT', 'COTIUSDT', 'BATUSDT', 'OGNUSDT', 'THETAUSDT', 'EGLDUSDT', 
    'ZILUSDT', 'SNXUSDT', 'HOTUSDT', 'APTUSDT', 'DYDXUSDT', 'RENDERUSDT', 
    'VETUSDT', 'LRCUSDT', 'TIAUSDT', 'ENJUSDT'
]

data_dir = '/Users/anders/projects/futures/data'
os.makedirs(data_dir, exist_ok=True)

# Use zenith_data_fetcher.py to ensure we have the data
# We assume zenith_data_fetcher.py is already configured to download to the data/ directory
# Let's verify if data exists for these assets
missing_assets = []
for asset in assets:
    if not os.path.exists(f"{data_dir}/{asset}_10m.csv"):
        missing_assets.append(asset)

if missing_assets:
    print(f"Missing data for: {missing_assets}. Fetching...")
    # Trigger fetch for missing assets (passing them as arguments if the script supports it)
    # For now, let's run the fetcher for the specific assets if we can, or just download them.
    for asset in missing_assets:
         subprocess.run(["python3", "/Users/anders/projects/futures/zenith_data_fetcher.py", "--ticker", asset])
else:
    print("All Top 16 asset data present.")
