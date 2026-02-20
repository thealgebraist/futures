import pandas as pd
import os

# Known ATH values (approximate)
ath_values = {
    'ICPUSDT': 750.0, 'FILUSDT': 237.0, 'FLOWUSDT': 46.0, 'ALGOUSDT': 3.5, 
    'VETUSDT': 0.28, 'IOTAUSDT': 5.6, 'ONTUSDT': 11.0, 'ZILUSDT': 0.25, 
    'HOTUSDT': 0.03, 'ONEUSDT': 0.38, 'CHZUSDT': 0.89, 'REQUSDT': 1.18, 
    'OGNUSDT': 3.4, 'RAYUSDT': 16.9, 'LRCUSDT': 3.8, 'SXPUSDT': 6.1, 
    'ENJUSDT': 4.8, 'THETAUSDT': 15.9, 'EGLDUSDT': 500.0, 'GRTUSDT': 2.8, 
    'CRVUSDT': 6.0, 'SNXUSDT': 28.0, 'DYDXUSDT': 27.0, 'HBARUSDT': 0.57, 
    'BATUSDT': 1.9, 'KAVAUSDT': 9.1, 'QTUMUSDT': 35.0, 'IOSTUSDT': 0.13, 
    'OMGUSDT': 28.0, 'FUNUSDT': 0.19, 'DGBUSDT': 0.18, 'SCUSDT': 0.11,
    'ANKRUSDT': 0.22, 'STORJUSDT': 4.0, 'KNCUSDT': 5.5, 'BANDUSDT': 23.0,
    'AUDIOVUSDT': 5.0, 'LDOUSDT': 7.3, 'AAVEUSDT': 660.0, 'MKRUSDT': 6300.0,
    'UNIUSDT': 45.0, 'LINKUSDT': 52.0, 'NEARUSDT': 20.0, 'ADAUSDT': 3.1,
    'SOLUSDT': 260.0, 'AVAXUSDT': 146.0, 'DOTUSDT': 55.0, 'TRXUSDT': 0.3, # TRX actually near ATH now
    'XRPUSDT': 3.8, 'MATICUSDT': 2.9, 'BCHUSDT': 4300.0, 'LTCUSDT': 412.0
}

distressed = []
data_dir = '/Users/anders/projects/futures/data/audit'

for ticker, ath in ath_values.items():
    csv_path = f"{data_dir}/{ticker}/10m.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        current_price = df['close'].iloc[-1]
        ratio = ath / current_price
        if ratio > 10.0:
            distressed.append({
                'ticker': ticker,
                'ath': ath,
                'current': current_price,
                'ratio': ratio
            })

df_distressed = pd.DataFrame(distressed).sort_values(by='ratio', ascending=False)
print(df_distressed.head(32))
df_distressed.head(32).to_csv('/Users/anders/projects/futures/distressed_32_ecoins.csv', index=False)
