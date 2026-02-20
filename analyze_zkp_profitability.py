
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# --- Configuration & Data ---
CONFIG = {
    "gpus": {
        "RTX_4090": {"cost_hr": 0.28, "aleo_hash": 1.4e6, "taiko_proof_sec": 15, "cpu_cores_req": 4, "ram_gb_req": 16},
        "RTX_5090": {"cost_hr": 0.37, "aleo_hash": 1.8e6, "taiko_proof_sec": 10, "cpu_cores_req": 6, "ram_gb_req": 24},
        "A100_80G": {"cost_hr": 1.10, "aleo_hash": 2.2e6, "taiko_proof_sec": 8, "cpu_cores_req": 8, "ram_gb_req": 80},
        "H100": {"cost_hr": 2.40, "aleo_hash": 3.5e6, "taiko_proof_sec": 4, "cpu_cores_req": 16, "ram_gb_req": 80},
    },
    "networks": {
        "Aleo": {"total_hash": 4.1e12, "block_reward": 23.7 * 0.9, "block_time": 10, "token_price": 12.0},
        "Taiko": {"avg_fee_usd": 0.05, "success_prob_base": 0.3, "latency_penalty_coeff": 0.05}
    },
    "server_limits": {"cpu_cores": 32, "ram_gb": 128}
}

def solve_allocation():
    names = list(CONFIG["gpus"].keys())
    hourly_profits = []
    for name in names:
        spec = CONFIG["gpus"][name]
        aleo_reward = (spec["aleo_hash"] / CONFIG["networks"]["Aleo"]["total_hash"]) * (3600 / CONFIG["networks"]["Aleo"]["block_time"]) * CONFIG["networks"]["Aleo"]["block_reward"] * CONFIG["networks"]["Aleo"]["token_price"]
        taiko_reward = (3600 / (spec["taiko_proof_sec"] * 2)) * CONFIG["networks"]["Taiko"]["avg_fee_usd"] * (CONFIG["networks"]["Taiko"]["success_prob_base"] * (1 - CONFIG["networks"]["Taiko"]["latency_penalty_coeff"] * spec["taiko_proof_sec"]))
        hourly_profits.append(aleo_reward + taiko_reward - spec["cost_hr"])

    # Objective: Minimize -Profit (Maximize Profit)
    c = -np.array(hourly_profits)
    A = [
        [CONFIG["gpus"][name]["cpu_cores_req"] for name in names],
        [CONFIG["gpus"][name]["ram_gb_req"] for name in names]
    ]
    b = [CONFIG["server_limits"]["cpu_cores"], CONFIG["server_limits"]["ram_gb"]]
    
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None) for _ in names], method='highs')
    # Rounding to simulate integer choice for simplicity in small scale
    allocation = {names[i]: int(np.floor(res.x[i])) for i in range(len(names))}
    return allocation

def monte_carlo_profit_distribution(selected_gpus, iterations=10000):
    results = []
    for _ in range(iterations):
        hourly_profit = 0
        for name, count in selected_gpus.items():
            if count <= 0: continue
            spec = CONFIG["gpus"][name]
            p_win = (spec["aleo_hash"] / CONFIG["networks"]["Aleo"]["total_hash"])
            total_blocks = 3600 / CONFIG["networks"]["Aleo"]["block_time"]
            blocks_won = np.random.poisson(p_win * total_blocks * count)
            aleo_rev = blocks_won * CONFIG["networks"]["Aleo"]["block_reward"] * CONFIG["networks"]["Aleo"]["token_price"]
            taiko_attempts = (3600 / (spec["taiko_proof_sec"] * 2)) * count
            success_prob = CONFIG["networks"]["Taiko"]["success_prob_base"] * np.exp(-0.02 * spec["taiko_proof_sec"])
            successes = np.random.binomial(int(taiko_attempts), success_prob)
            taiko_rev = successes * CONFIG["networks"]["Taiko"]["avg_fee_usd"]
            hourly_profit += (aleo_rev + taiko_rev) - (spec["cost_hr"] * count)
        results.append(hourly_profit)
    return np.array(results)

if __name__ == "__main__":
    allocation = solve_allocation()
    dist = monte_carlo_profit_distribution(allocation)
    summary = {
        "mean": float(np.mean(dist)),
        "std": float(np.std(dist)),
        "prob_profit": float(np.mean(dist > 0)),
        "allocation": allocation
    }
    print(summary)
    pd.Series(summary).to_json("profit_sim_results.json")
