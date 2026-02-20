import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# --- Model Definition ---
class FFNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=512, output_dim=4):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def estimate_garch(returns, omega=1e-6, alpha=0.1, beta=0.8):
    n = len(returns)
    sigmas = np.zeros(n)
    sigmas[0] = np.var(returns)
    for t in range(1, n):
        sigmas[t] = omega + alpha * (returns[t-1]**2) + beta * sigmas[t-1]
    return np.sqrt(sigmas)

def run_dkk_audit():
    coin = 'SOLUSDT'
    csv_path = f'data/audit/{coin}/1m.csv'
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    split_date = pd.to_datetime('2026-02-18 00:00:00')
    train_df = df[df['timestamp'] < split_date]
    test_df = df[df['timestamp'] >= split_date]
    
    train_prices = train_df['close'].values
    test_prices = test_df['close'].values
    
    scaler = StandardScaler()
    p_train_scaled = scaler.fit_transform(train_prices.reshape(-1, 1)).flatten()
    p_test_scaled = scaler.transform(test_prices.reshape(-1, 1)).flatten()
    
    input_steps = 16
    # Load the model we just "trained" in high speed for the last audit
    # Actually, for deterministic results in this report, I'll just re-run the simulation logic 
    # but I'll need a model state. Since the previous script didn't save the model, 
    # I'll re-train it VERY quickly (10s) or just simulate the signals we have in solusdt_trades.csv.
    
    # Better: Read solusdt_trades.csv and the full test_df to reconstruct the exact capital path.
    trades_df = pd.read_csv('solusdt_trades.csv')
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    dkk_rate = 7.01
    capital_usd = 100.0
    
    ledger = []
    
    # Reconstruct equity curve step by step
    # Zenith enters a trade at t, return is realized at t+1.
    # In the simple simulation, we stay in position for one interval (1 min).
    
    for i in range(len(test_prices) - 1):
        ts = test_df.iloc[i]['timestamp']
        is_trade = ts in trades_df['timestamp'].values
        
        pos = -1 if is_trade else 0
        
        if pos == -1:
            # Entry (Opening Short)
            # We "sell" SOL worth our entire capital
            sell_usd = capital_usd
            sell_dkk = sell_usd * dkk_rate
            
            # Profit calculation: capital * (1 + actual_ret * pos)
            actual_ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]
            old_capital = capital_usd
            capital_usd *= (1 + actual_ret * pos)
            
            # Exit (Closing Short)
            # The "buy back" value is what we paid to close the position
            # Profit = Sell Proceeds - Buy Back Cost
            # So, Buy Back Cost = Sell Proceeds - USD Profit
            usd_profit = capital_usd - old_capital
            buy_back_usd = sell_usd - usd_profit
            buy_back_dkk = buy_back_usd * dkk_rate
            
            ledger.append({
                'timestamp': ts,
                'action': 'OPEN SHORT (SELL SOL)',
                'price_usd': test_prices[i],
                'amount_usd': sell_usd,
                'amount_dkk': sell_dkk,
                'balance_dkk': old_capital * dkk_rate
            })
            
            ledger.append({
                'timestamp': test_df.iloc[i+1]['timestamp'],
                'action': 'CLOSE SHORT (BUY SOL)',
                'price_usd': test_prices[i+1],
                'amount_usd': buy_back_usd,
                'amount_dkk': buy_back_dkk,
                'balance_dkk': capital_usd * dkk_rate
            })
            
    ledger_df = pd.DataFrame(ledger)
    ledger_df.to_csv('solusdt_dkk_ledger.csv', index=False)
    print("DKK Ledger generated.")

if __name__ == "__main__":
    run_dkk_audit()
