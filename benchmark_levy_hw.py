import subprocess
import os
import tempfile
import shutil

def run_hw_benchmarks():
    stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
    ecoins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    symbols = stocks + ecoins
    
    # 15s per run * 2 backends * 8 symbols = 240s (4 mins)
    # 1 min for compilation and overhead.
    train_duration = 15 
    
    build_dir = tempfile.mkdtemp()
    print(f"Build directory: {build_dir}")
    
    try:
        src_dir = '/Users/anders/projects/futures/src'
        accel_bin = os.path.join(build_dir, 'levy_accel')
        neon_bin = os.path.join(build_dir, 'levy_neon')
        
        print("Compiling Accelerate...")
        subprocess.run(['clang++', '-std=c++2b', '-O3', '-framework', 'Accelerate', 
                        '-I', src_dir, '-o', accel_bin, os.path.join(src_dir, 'levy_trainer_accelerate.cpp')], check=True)
        
        print("Compiling NEON...")
        subprocess.run(['clang++', '-std=c++2b', '-O3', '-march=armv8-a+simd', 
                        '-I', src_dir, '-o', neon_bin, os.path.join(src_dir, 'levy_trainer_neon.cpp')], check=True)
        
        results = []
        
        for s in symbols:
            if s in stocks: path = f'/Users/anders/projects/futures/data/signature_experiment/{s}_4y.csv'
            elif s == 'BTCUSDT': path = '/Users/anders/projects/futures/data/btc_experiment/btc_1m_3mo.csv'
            elif s == 'SOLUSDT': path = '/Users/anders/projects/futures/data/sol_experiment/sol_10m_1y.csv'
            else: path = f'/Users/anders/projects/futures/data/ecoins_2y/{s}_10m_2y.csv'
            
            if not os.path.exists(path):
                print(f"Skipping {s}, missing {path}")
                continue
                
            print(f"Running {s} (15s each)...")
            
            # Accelerate
            r_accel = subprocess.run([accel_bin, path, s, str(train_duration)], capture_output=True, text=True)
            if r_accel.stdout: 
                print(f"  {r_accel.stdout.strip()}")
                results.append(r_accel.stdout.strip())
            
            # NEON
            r_neon = subprocess.run([neon_bin, path, s, str(train_duration)], capture_output=True, text=True)
            if r_neon.stdout: 
                print(f"  {r_neon.stdout.strip()}")
                results.append(r_neon.stdout.strip())
            
        with open('/Users/anders/projects/futures/levy_hw_results.txt', 'w') as f:
            for r in results: f.write(r + '\n')
            
    finally:
        shutil.rmtree(build_dir)
        print("Build directory cleaned.")

if __name__ == "__main__":
    run_hw_benchmarks()
