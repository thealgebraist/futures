import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import os

def analyze_gp_derivatives(ticker, days_to_sample=30):
    print(f"Analyzing GP derivatives for {ticker}...")
    
    # Load data for the specific ticker
    path = ""
    if os.path.exists(f"data/returns_2y/{ticker}_10m.csv"):
        path = f"data/returns_2y/{ticker}_10m.csv"
    elif os.path.exists(f"data/ecoins_2y/{ticker}_10m_2y.csv"):
        path = f"data/ecoins_2y/{ticker}_10m_2y.csv"
    elif os.path.exists(f"data/audit/{ticker}/10m.csv"):
        path = f"data/audit/{ticker}/10m.csv"
    
    if not path:
        print(f"  No data found for {ticker}. Skipping GP analysis.")
        return None, None
        
    df = pd.read_csv(path)
    df.columns = ['timestamp', 'close'] # Assuming standard columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Select a continuous block of data for GP fitting (e.g., 30 days)
    # Ensure there are enough data points
    if len(df) < days_to_sample * 24 * 6: # 24 hours * 6 10-min intervals
        print(f"  Not enough data for {ticker} for {days_to_sample} days. Skipping GP analysis.")
        return None, None

    # Pick a random starting point that allows for days_to_sample
    max_start_idx = len(df) - (days_to_sample * 24 * 6)
    if max_start_idx <= 0:
        start_idx = 0
    else:
        start_idx = np.random.randint(0, max_start_idx)
    
    sample_df = df.iloc[start_idx : start_idx + (days_to_sample * 24 * 6)]
    
    X = (sample_df.index.asi8.reshape(-1, 1) - sample_df.index.asi8[0]) / (1e9 * 60 * 10) # Time in 10-min intervals
    y = np.log(sample_df['close'].values) # Log prices for better stationarity
    
    # Gaussian Process setup
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
    
    print(f"  Fitting GP for {ticker} over {len(y)} samples...")
    try:
        gp.fit(X, y)
    except Exception as e:
        print(f"  GP fitting failed for {ticker}: {e}. Skipping.")
        return None, None

    # Predict mean and standard deviation for the entire sampled range
    X_pred = X # Predict on the same points
    y_mean, y_std = gp.predict(X_pred, return_std=True)

    # Numerical derivatives of the GP mean function
    # First derivative (approximate log-returns)
    gp_log_returns = np.diff(y_mean)
    
    # Second derivative (approximate derivative of log-returns)
    gp_second_diffs = np.diff(gp_log_returns)

    # Plot GP fit and derivatives for visual inspection (for BTC)
    if ticker == 'BTCUSDT':
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(sample_df.index, y, 'r.', markersize=5, label='Actual Log Prices')
        plt.plot(sample_df.index, y_mean, 'b-', label='GP Mean Fit')
        plt.fill_between(sample_df.index, y_mean - y_std, y_mean + y_std, alpha=0.2, color='blue', label='GP Std Dev')
        plt.title(f'{ticker} Log Prices with GP Fit ({days_to_sample} Days)')
        plt.ylabel('Log Price')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(sample_df.index[:-1], gp_log_returns, 'g-', label='GP Log Returns (1st Deriv)')
        plt.title('GP-Derived Log Returns')
        plt.ylabel('Log Return')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(sample_df.index[:-2], gp_second_diffs, 'm-', label='GP Second Diffs (2nd Deriv)')
        plt.title('GP-Derived Second Differences')
        plt.xlabel('Time')
        plt.ylabel('Second Diff')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'gp_derivatives_{ticker}.png')
        plt.close()
        print(f"  Saved GP derivatives plot for {ticker} to gp_derivatives_{ticker}.png")

    return gp_log_returns, gp_second_diffs

def main():
    assets_to_process = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT',
        'TRXUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'LTCUSDT', 'UNIUSDT', 'BCHUSDT',
        'NEARUSDT', 'FETUSDT', 'APTUSDT', 'ICPUSDT', 'STXUSDT', 'OPUSDT', 'FILUSDT',
        'XLMUSDT', 'AAVEUSDT', 'GRTUSDT', 'HBARUSDT', 'KASUSDT', 'SUIUSDT', 'ARBUSDT',
        'RENDERUSDT', 'PEPEUSDT', 'WLDUSDT', 'IMXUSDT', 'INJUSDT', 'TIAUSDT', 'LDOUSDT',
        'MKRUSDT', 'KNCUSDT', 'CRVUSDT', 'MANAUSDT', 'EGLDUSDT', 'ENJUSDT', 'CHZUSDT',
        'ZILUSDT', 'SNXUSDT', 'HOTUSDT', 'DYDXUSDT', 'FLOWUSDT', 'IOSTUSDT', 'IOTAUSDT',
        'QTUMUSDT', 'RAYUSDT', 'SXPUSDT', 'THETAUSDT', 'VETUSDT', 'SCUSDT', 'ONDOUSDT',
        'ONEUSDT', 'ONTUSDT', 'SUSHIUSDT', 'ALGOUSDT', 'DGBUSDT', 'ALPHAUSDT', 'ANKRUSDT',
        'GLMUSDT'
    ]

    all_gp_log_returns = []
    all_gp_second_diffs = []
    
    # Process a subset for testing, GP is computationally intensive
    # Take the first 10 assets for GP analysis
    for ticker in assets_to_process[:10]:
        gp_lr, gp_sd = analyze_gp_derivatives(ticker)
        if gp_lr is not None:
            all_gp_log_returns.extend(gp_lr)
            all_gp_second_diffs.extend(gp_sd)
            
    # Save GP-derived data for fitting
    gp_returns_df = pd.DataFrame({'return': all_gp_log_returns})
    gp_returns_df.to_csv('data/all_gp_log_returns.csv', index=False)
    
    gp_second_diffs_df = pd.DataFrame({'second_diff': all_gp_second_diffs})
    gp_second_diffs_df.to_csv('data/all_gp_second_diffs.csv', index=False)
    
    print("Saved aggregated GP-derived log returns to data/all_gp_log_returns.csv")
    print("Saved aggregated GP-derived second differences to data/all_gp_second_diffs.csv")

if __name__ == "__main__":
    main()
