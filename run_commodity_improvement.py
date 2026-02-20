import subprocess
import os
import random

def run_improvement_loop():
    symbols = ['EURUSD=X', 'CORN', 'WEAT', 'DBA']
    
    # Initial hyperparameters
    states = {s: {"lr": 0.001, "threshold": 0.0005} for s in symbols}
    
    # Compile
    subprocess.run(['clang++', '-std=c++2b', '-O3', '-framework', 'Accelerate', '-o', 'commodity_trainer', 'src/commodity_trainer.cpp'], check=True)
    
    history = []
    
    for round_idx in range(8):
        print(f"--- Round {round_idx + 1} of 8 ---")
        round_results = {}
        
        for i, symbol in enumerate(symbols):
            s_state = states[symbol]
            seed = random.randint(0, 1000000)
            
            # Run C++ trainer
            # argv: [1]col_idx, [2]lr, [3]threshold, [4]seed
            cmd = [
                './commodity_trainer', 
                str(i), 
                f"{s_state['lr']:.8f}", 
                f"{s_state['threshold']:.8f}", 
                str(seed)
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            try:
                final_cap = float(process.stdout.strip())
            except:
                final_cap = 0.0
                
            print(f"  {symbol}: Final Capital ${final_cap:.2f} | LR: {s_state['lr']:.6f} | Thresh: {s_state['threshold']:.6f}")
            round_results[symbol] = final_cap
            
            # IMPROVEMENT LOGIC
            if final_cap > 100.0:
                # Success: reinforce parameters but maybe tighten threshold
                s_state['lr'] *= 1.1
                s_state['threshold'] *= 1.05
            else:
                # Failure: reduce learning rate and lower threshold to capture more signals
                s_state['lr'] *= 0.7
                s_state['threshold'] *= 0.9
                
        history.append((round_idx + 1, round_results))
        
    # Write summary
    with open("commodity_improvement_results.txt", "w") as f:
        f.write("Round " + " ".join(symbols) + "\n")
        for h in history:
            line = f"{h[0]} " + " ".join([f"{h[1][s]:.2f}" for s in symbols]) + "\n"
            f.write(line)

if __name__ == "__main__":
    run_improvement_loop()
