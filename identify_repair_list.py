import os
import pandas as pd

assets = [
    'PEPEUSDT', 'BONKUSDT', 'SHIBUSDT', 'GRTUSDT', 'ALGOUSDT', 'MANAUSDT', 'HBARUSDT', 'DOGEUSDT', 
    'POLUSDT', 'APEUSDT', 'XLMUSDT', 'SUSHIUSDT', 'ONDOUSDT', 'ADAUSDT', 'TRXUSDT', 'WLDUSDT', 
    'XTZUSDT', 'SUIUSDT', 'FILUSDT', 'DOTUSDT', 'NEARUSDT', 'TONUSDT', 'XRPUSDT', 'ICPUSDT', 
    'ATOMUSDT', 'UNIUSDT', 'LINKUSDT', 'AVAXUSDT', 'HYPEUSDT', 'LTCUSDT', 'SOLUSDT', 'AAVEUSDT', 
    'ETHUSDT', 'BTCUSDT'
]

data_dir = '/Users/anders/projects/futures/data/audit'
to_fix = []

for ticker in assets:
    path = os.path.join(data_dir, ticker, '10m.csv')
    if not os.path.exists(path):
        to_fix.append(ticker)
    else:
        try:
            df = pd.read_csv(path, nrows=1)
            if len(df.columns) < 6:
                to_fix.append(ticker)
        except:
            to_fix.append(ticker)

print(to_fix)
