import numpy as np
import pandas as pd

def simulate_trading_month(initial_capital, target_profit, risk_per_trade_pct, win_rate, rr_ratio, num_trades_per_day=3, days=20):
    capital = initial_capital
    risk_amount = initial_capital * risk_per_trade_pct
    cost_per_trade = 1.50 
    num_trades = days * num_trades_per_day
    
    for _ in range(num_trades):
        if capital <= initial_capital * 0.8:
            return capital
            
        if np.random.random() < win_rate:
            capital += (risk_amount * rr_ratio) - cost_per_trade
        else:
            capital -= (risk_amount + cost_per_trade)
        
    return capital

def run_monte_carlo(initial_capital, target_profit, risk_per_trade_pct, win_rate, rr_ratio, iterations=10000):
    capitals = [simulate_trading_month(initial_capital, target_profit, risk_per_trade_pct, win_rate, rr_ratio) for _ in range(iterations)]
    capitals = np.array(capitals)
    
    success = np.mean(capitals >= initial_capital + target_profit)
    ruin = np.mean(capitals <= initial_capital * 0.8)
    avg_final = np.mean(capitals)
    
    return success, ruin, avg_final

def main():
    target = 100
    # capital, risk, wr, rr
    configs = [
        (500, 0.02, 0.5, 1.5),
        (500, 0.05, 0.5, 1.5),
        (1000, 0.01, 0.5, 1.5),
        (1000, 0.02, 0.5, 1.5),
        (1000, 0.02, 0.4, 2.0),
        (2000, 0.01, 0.5, 1.5),
        (5000, 0.005, 0.5, 1.5),
    ]
    
    results = []
    for cap, risk, wr, rr in configs:
        p_success, p_ruin, avg_final = run_monte_carlo(cap, target, risk, wr, rr)
        results.append({
            "Capital": cap,
            "Risk_Pct": risk * 100,
            "Win_Rate": wr,
            "RR": rr,
            "Prob_Success": p_success,
            "Prob_Ruin": p_ruin,
            "Exp_Value": avg_final - cap
        })
        
    df = pd.DataFrame(results)
    df.to_csv("/Users/anders/projects/futures/simulation_results.csv", index=False)
    
    # Generate LaTeX table snippet
    with open("/Users/anders/projects/futures/results_table.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Capital & Risk\\% & WR & RR & P(Success) & P(Ruin) & Exp. Profit \\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            f.write(f"{int(row['Capital'])} & {row['Risk_Pct']:.1f} & {row['Win_Rate']:.1f} & {row['RR']:.1f} & {row['Prob_Success']:.2f} & {row['Prob_Ruin']:.2f} & \\${row['Exp_Value']:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

if __name__ == "__main__":
    main()
