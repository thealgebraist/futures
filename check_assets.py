import os

assets = [
    'PEPEUSDT', 'BONKUSDT', 'SHIBUSDT', 'GRTUSDT', 'ALGOUSDT', 'MANAUSDT', 'HBARUSDT', 'DOGEUSDT', 
    'POLUSDT', 'APEUSDT', 'XLMUSDT', 'SUSHIUSDT', 'ONDOUSDT', 'ADAUSDT', 'TRXUSDT', 'WLDUSDT', 
    'XTZUSDT', 'SUIUSDT', 'FILUSDT', 'DOTUSDT', 'NEARUSDT', 'TONUSDT', 'XRPUSDT', 'ICPUSDT', 
    'ATOMUSDT', 'UNIUSDT', 'LINKUSDT', 'AVAXUSDT', 'HYPEUSDT', 'LTCUSDT', 'SOLUSDT', 'AAVEUSDT', 
    'ETHUSDT', 'BTCUSDT'
]

data_dir = '/Users/anders/projects/futures/data/audit'
missing = []
present = []

for ticker in assets:
    path = os.path.join(data_dir, ticker, '10m.csv')
    if os.path.exists(path):
        present.append(ticker)
    else:
        missing.append(ticker)

print(f"Present: {len(present)}")
print(f"Missing: {len(missing)}")
print(f"Missing list: {missing}")
