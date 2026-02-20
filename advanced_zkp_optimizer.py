import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
import time
import json

# 1. Detailed Hardware Database (Top 16 GPUs across Vast.ai, RunPod, Lambda, Hyperstack, FluidStack)
# Metrics: VRAM(GB), TFLOPS(FP32), CUDA Cores, Memory Bandwidth(GB/s), L2 Cache(MB), ZK_Hashrate(Mh/s), L2_Proof_Sec
gpus = {
    "RTX_5090": {"cost_hr": 0.37, "vram": 32, "tflops": 115, "cores": 24576, "mem_bw": 1792, "l2_cache": 128, "zk_hash": 1.85, "l2_sec": 10},
    "RTX_4090": {"cost_hr": 0.28, "vram": 24, "tflops": 82.6, "cores": 16384, "mem_bw": 1008, "l2_cache": 72, "zk_hash": 1.40, "l2_sec": 15},
    "RTX_5080": {"cost_hr": 0.22, "vram": 16, "tflops": 80,  "cores": 10752, "mem_bw": 1024, "l2_cache": 64, "zk_hash": 1.10, "l2_sec": 18},
    "RTX_4080": {"cost_hr": 0.18, "vram": 16, "tflops": 48.7, "cores": 9728,  "mem_bw": 716,  "l2_cache": 64, "zk_hash": 0.85, "l2_sec": 22},
    "RTX_3090": {"cost_hr": 0.20, "vram": 24, "tflops": 35.6, "cores": 10496, "mem_bw": 936,  "l2_cache": 6,  "zk_hash": 0.75, "l2_sec": 25},
    "H200_141G": {"cost_hr": 2.50, "vram": 141,"tflops": 1979,"cores": 16896, "mem_bw": 4800, "l2_cache": 50, "zk_hash": 4.20, "l2_sec": 3},
    "H100_80G":  {"cost_hr": 1.80, "vram": 80, "tflops": 1513,"cores": 14592, "mem_bw": 3350, "l2_cache": 50, "zk_hash": 3.50, "l2_sec": 4},
    "A100_80G":  {"cost_hr": 1.10, "vram": 80, "tflops": 312, "cores": 6912,  "mem_bw": 1935, "l2_cache": 40, "zk_hash": 2.20, "l2_sec": 8},
    "A100_40G":  {"cost_hr": 0.85, "vram": 40, "tflops": 156, "cores": 6912,  "mem_bw": 1555, "l2_cache": 40, "zk_hash": 1.60, "l2_sec": 11},
    "L40S":      {"cost_hr": 0.90, "vram": 48, "tflops": 91.6, "cores": 18176, "mem_bw": 864,  "l2_cache": 96, "zk_hash": 1.90, "l2_sec": 9},
    "RTX_6000Ada":{"cost_hr": 1.05,"vram": 48, "tflops": 91.1, "cores": 18176, "mem_bw": 960,  "l2_cache": 96, "zk_hash": 1.95, "l2_sec": 9},
    "RTX_A6000": {"cost_hr": 0.65, "vram": 48, "tflops": 38.7, "cores": 10752, "mem_bw": 768,  "l2_cache": 6,  "zk_hash": 1.05, "l2_sec": 16},
    "V100_32G":  {"cost_hr": 0.40, "vram": 32, "tflops": 14.1, "cores": 5120,  "mem_bw": 900,  "l2_cache": 6,  "zk_hash": 0.50, "l2_sec": 30},
    "RTX_3080":  {"cost_hr": 0.15, "vram": 10, "tflops": 29.8, "cores": 8704,  "mem_bw": 760,  "l2_cache": 5,  "zk_hash": 0.55, "l2_sec": 35},
    "RTX_4070Ti":{"cost_hr": 0.16, "vram": 12, "tflops": 40.1, "cores": 7680,  "mem_bw": 504,  "l2_cache": 48, "zk_hash": 0.65, "l2_sec": 28},
    "T4_16G":    {"cost_hr": 0.10, "vram": 16, "tflops": 8.1,  "cores": 2560,  "mem_bw": 320,  "l2_cache": 4,  "zk_hash": 0.15, "l2_sec": 90},
}

ALEO_NET_HASH = 4.1e12
ALEO_REWARD_24H = 23.7 * (24 * 360) * 12.0 
L2_FEE_USD = 0.05
L2_SUCCESS_BASE = 0.3

# Derive expected 24h revenue per GPU combining Aleo + Taiko L2
for k, v in gpus.items():
    # Aleo PoSW Revenue
    p_win = v["zk_hash"] * 1e6 / ALEO_NET_HASH
    aleo_rev = p_win * ALEO_REWARD_24H
    
    # Taiko / zkSync L2 Proving Revenue
    # proofs attempted per day = (24 * 3600) / (l2_sec * 2) 
    l2_attempts = (24 * 3600) / (v["l2_sec"] * 2)
    l2_prob = L2_SUCCESS_BASE * np.exp(-0.02 * v["l2_sec"])
    l2_rev = l2_attempts * l2_prob * L2_FEE_USD
    
    v["rev_24h"] = aleo_rev + l2_rev
    v["cost_24h"] = v["cost_hr"] * 24
    v["profit_24h"] = v["rev_24h"] - v["cost_24h"]

names = list(gpus.keys())
n_gpus = len(names)

rev_24h = np.array([gpus[n]["rev_24h"] for n in names])
cost_24h = np.array([gpus[n]["cost_24h"] for n in names])
profit_24h = rev_24h - cost_24h

# Constraints
BUDGET_24H = 150.0 # Max $150 spend per day
MAX_SERVERS = 10

results = {}

# --- 2. Linear Programming (LP) ---
# Minimize -Profit
c_lp = -profit_24h
A_ub = [cost_24h, np.ones(n_gpus)]
b_ub = [BUDGET_24H, MAX_SERVERS]
bounds = [(0, None) for _ in names]

start_time = time.time()
res_lp = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'time_limit': 120})
lp_time = time.time() - start_time
alloc_lp = {names[i]: float(res_lp.x[i]) for i in range(n_gpus) if res_lp.x[i] > 1e-4}
results['LP'] = {"profit": -res_lp.fun, "time": lp_time, "alloc": alloc_lp}

# --- 3. Mixed Integer Linear Programming (MILP) ---
try:
    from scipy.optimize import milp, LinearConstraint
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    integrality = np.ones(n_gpus)

    start_time = time.time()
    res_milp = milp(c=c_lp, constraints=constraints, integrality=integrality, bounds=None, options={'time_limit': 120})
    milp_time = time.time() - start_time
    alloc_milp = {names[i]: int(np.round(res_milp.x[i])) for i in range(n_gpus) if res_milp.x[i] > 1e-4}
    results['MILP'] = {"profit": -res_milp.fun, "time": milp_time, "alloc": alloc_milp}
except ImportError:
    # Fallback to rounding LP if MILP not available
    results['MILP'] = {"profit": -res_lp.fun, "time": lp_time, "alloc": {k: int(v) for k,v in alloc_lp.items()}}

# --- 4. Convex / Non-Linear Programming (CP) ---
def objective_cp(x):
    # - (Revenue * sqrt(x) - Cost * x)
    rev = np.sum(rev_24h * x * (1.0 / (1.0 + 0.05 * x))) 
    cost = np.sum(cost_24h * x)
    return -(rev - cost)

def constraint_budget(x):
    return BUDGET_24H - np.sum(cost_24h * x)

def constraint_servers(x):
    return MAX_SERVERS - np.sum(x)

bounds_cp = [(0, MAX_SERVERS) for _ in range(n_gpus)]
x0 = np.ones(n_gpus)

start_time = time.time()
res_cp = minimize(objective_cp, x0, method='SLSQP', bounds=bounds_cp, 
                  constraints=[{'type': 'ineq', 'fun': constraint_budget},
                               {'type': 'ineq', 'fun': constraint_servers}],
                  options={'maxiter': 1000}) 
cp_time = time.time() - start_time
alloc_cp = {names[i]: float(res_cp.x[i]) for i in range(n_gpus) if res_cp.x[i] > 1e-2}
results['Convex'] = {"profit": -res_cp.fun, "time": cp_time, "alloc": alloc_cp}

# --- 5. Monte Carlo Simulation for Probability Distribution (using MILP allocation) ---
def simulate_24h_profit(allocation, iterations=20000):
    profits = []
    for _ in range(iterations):
        daily_profit = 0
        for name, count in allocation.items():
            if count <= 0: continue
            spec = gpus[name]
            
            # Stochastic block wins Aleo
            p_win = (spec["zk_hash"] * 1e6 / ALEO_NET_HASH)
            blocks_won = np.random.poisson(p_win * (24*360) * count) 
            rev_aleo = blocks_won * 23.7 * 12.0
            
            # Stochastic L2 wins Taiko
            l2_attempts = (24 * 3600) / (spec["l2_sec"] * 2) * count
            success_prob = L2_SUCCESS_BASE * np.exp(-0.02 * spec["l2_sec"])
            successes = np.random.binomial(int(l2_attempts), success_prob)
            rev_l2 = successes * L2_FEE_USD
            
            daily_profit += (rev_aleo + rev_l2) - (spec["cost_hr"] * 24 * count)
            
        profits.append(daily_profit)
    return np.array(profits)

dist = simulate_24h_profit(results['MILP']['alloc'])
results['MonteCarlo'] = {
    "mean_profit_24h": float(np.mean(dist)),
    "std_dev": float(np.std(dist)),
    "prob_profit_gt_0": float(np.mean(dist > 0)),
    "prob_profit_gt_50": float(np.mean(dist > 50)),
    "p5_profit": float(np.percentile(dist, 5)),
    "p95_profit": float(np.percentile(dist, 95))
}

with open("advanced_opt_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Optimization complete.")
