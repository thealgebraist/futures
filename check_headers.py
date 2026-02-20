import os

assets = [
    'PEPEUSDT', 'BONKUSDT', 'SHIBUSDT', 'GRTUSDT', 'ALGOUSDT', 'MANAUSDT', 'HBARUSDT', 'DOGEUSDT', 
    'POLUSDT', 'APEUSDT', 'XLMUSDT', 'SUSHIUSDT', 'ONDOUSDT', 'ADAUSDT', 'TRXUSDT', 'WLDUSDT', 
    'XTZUSDT', 'SUIUSDT', 'FILUSDT', 'DOTUSDT', 'NEARUSDT', 'TONUSDT', 'XRPUSDT', 'ICPUSDT', 
    'ATOMUSDT', 'UNIUSDT', 'LINKUSDT', 'AVAXUSDT', 'HYPEUSDT', 'LTCUSDT', 'SOLUSDT', 'AAVEUSDT', 
    'ETHUSDT', 'BTCUSDT'
]

print("Ticker,Header")
for ticker in assets:
    path = f"/Users/anders/projects/futures/data/audit/{ticker}/10m.csv"
    if os.path.exists(path):
        with open(path, 'r') as f:
            header = f.readline().strip()
            print(f"{ticker},{header}")
    else:
        print(f"{ticker},MISSING")
