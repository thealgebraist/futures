import subprocess
import os
import pandas as pd

def run_benchmarks():
    stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
    ecoins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    symbols = stocks + ecoins
    
    # Compile
    print("Compiling Accelerate version...")
    subprocess.run(['clang++', '-std=c++2b', '-O3', '-framework', 'Accelerate', '-o', 'levy_accel', 'src/levy_trainer_accelerate.cpp'], check=True)
    print("Compiling NEON version...")
    subprocess.run(['clang++', '-std=c++2b', '-O3', '-march=armv8-a+simd', '-o', 'levy_neon', 'src/levy_trainer_neon.cpp'], check=True)
    
    results = []
    
    for s in symbols:
        # Determine path
        if s in stocks:
            path = f'data/signature_experiment/{s}_4y.csv'
        elif s == 'BTCUSDT':
            path = 'data/btc_experiment/btc_1m_3mo.csv'
        elif s == 'SOLUSDT':
            path = 'data/sol_experiment/sol_10m_1y.csv'
        else:
            path = f'data/ecoins_2y/{s}_10m_2y.csv'
            
        if not os.path.exists(path):
            print(f"Skipping {s}, path not found: {path}")
            continue
            
        print(f"Benchmarking {s}...")
        
        # Run Accel
        res_accel = subprocess.run(['./levy_accel', path, s], capture_output=True, text=True)
        if res_accel.stdout:
            results.append(res_accel.stdout.strip())
            
        # Run NEON
        res_neon = subprocess.run(['./levy_neon', path, s], capture_output=True, text=True)
        if res_neon.stdout:
            results.append(res_neon.stdout.strip())
            
    with open('levy_benchmark_results.txt', 'w') as f:
        for r in results:
            f.write(r + '\n')
            
if __name__ == "__main__":
    run_benchmarks()
