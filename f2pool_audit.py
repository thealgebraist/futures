import requests
import json
import sys
import time
from datetime import datetime

API_URL = "https://api.f2pool.com/v2"
API_SECRET = "b1qke9ho23bhpgo3z098juwnd57ievgjiatxc5ochtrwhgxlgm1rpbu5v9zpnuql"
USER_NAME = "anders2026"
CURRENCY = "aleo"

def query_f2pool(endpoint, payload):
    headers = {"Content-Type": "application/json", "F2P-API-SECRET": API_SECRET}
    url = f"{API_URL}/{endpoint}"
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return response.json() if response.status_code == 200 else None
    except: return None

def draw_graph(history, width=60, height=10):
    if not history: return "No historical data available."
    points = [p[1] for p in history]
    if not points: return "Graph: [Data empty]"
    
    max_h = max(points) if points else 1
    min_h = min(points) if points else 0
    
    graph = []
    for y in range(height, -1, -1):
        line = ""
        threshold = min_h + (max_h - min_h) * (y / height)
        for val in points[-width:]:
            if val >= threshold: line += "\033[1;32m#\033[0m"
            else: line += " "
        graph.append(f"{threshold/1e6:6.1f}M | {line}")
    return "\n".join(graph)

def main():
    print(f"\033[1;33m=== F2POOL ADVANCED AUDITOR v2: {USER_NAME} ===\033[0m")
    
    # 1. Fetch Global Account Status
    info = query_f2pool("hash_rate/info", {"currency": CURRENCY, "user_name": USER_NAME})
    if not info:
        print("\033[1;31m[ERR]\033[0m Could not reach F2Pool API. Check your Secret Key.")
        return

    # 2. Extract Metrics
    cur_hr = info["info"]["hash_rate"]
    avg_1h = info["info"]["h1_hash_rate"]
    avg_24h = info["info"]["h24_hash_rate"]
    stale_24h = info["info"]["h24_stale_hash_rate"]
    efficiency = (1.0 - (stale_24h / avg_24h)) * 100 if avg_24h > 0 else 0

    print(f"\n\033[1;37m[CURRENT]\033[0m {cur_hr/1e6:8.2f} Mh/s")
    print(f"\033[1;37m[AVG 1H ]\033[0m {avg_1h/1e6:8.2f} Mh/s")
    print(f"\033[1;37m[AVG 24H]\033[0m {avg_24h/1e6:8.2f} Mh/s")
    print(f"\033[1;37m[QUALITY]\033[0m {efficiency:8.2f}% Efficiency")

    # 3. Hashrate Graph (Historical)
    print("\n\033[1;36m--- 24-HOUR HASHRATE TREND ---\033[0m")
    # Fetch worker history for better resolution
    history_resp = query_f2pool("hash_rate/worker/history", {"currency": CURRENCY, "user_name": USER_NAME})
    if history_resp and "history" in history_resp:
        # Convert map to sorted list of points
        hist_data = sorted(history_resp["history"].items())
        # Points are usually [timestamp, hashrate]
        formatted_history = [[int(t), float(h)] for t, h in hist_data]
        print(draw_graph(formatted_history))
    else:
        print("Waiting for more data points to generate graph...")

    # 4. Revenue Projections
    balance = query_f2pool("assets/balance", {"currency": CURRENCY, "mining_user_name": USER_NAME})
    if balance:
        est_day = balance.get("balance_info", {}).get("estimated_today_income", 0)
        if est_day == 0 and avg_24h > 0:
            # Fallback estimation based on average hashrate if pool hasn't calculated yet
            est_day = (avg_24h / 1e6) * 0.15 # Placeholder coefficient for Aleo
        
        print("\n\033[1;35m--- REVENUE PROJECTIONS (ALEO) ---\033[0m")
        print(f"Next 24h: {est_day:8.4f} ALEO")
        print(f"Next 7d : {est_day*7:8.4f} ALEO")
        print(f"Next 30d: {est_day*30:8.4f} ALEO")

    print("\n\033[1;30mLast Update: " + datetime.now().strftime("%H:%M:%S") + "\033[0m")

if __name__ == "__main__":
    main()
