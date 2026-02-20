import subprocess
import os

def run_optimization():
    symbols = ['USO', 'UNG', 'XLE', 'XOM', 'CVX', 'NEE', 'CL=F', 'NG=F']
    
    # Initial parameters
    lr = 0.001
    
    # Compile
    subprocess.run(['clang++', '-std=c++2b', '-O3', '-framework', 'Accelerate', '-o', 'energy_trainer_v2', 'src/energy_trainer.cpp'], check=True)
    
    overall_results = []
    
    for round_idx in range(8):
        print(f"--- Round {round_idx + 1} of Improvements ---")
        round_pnl = []
        
        for i in range(8):
            symbol = symbols[i]
            # Run C++ trainer: argv[1]=col_idx, argv[2]=lr, argv[3]=round
            process = subprocess.run(['./energy_trainer_v2', str(i), f"{lr:.8f}", str(round_idx)], capture_output=True, text=True)
            
            try:
                final_capital = float(process.stdout.strip())
            except:
                final_capital = 0.0
                
            print(f"  {symbol}: Final Capital ${final_capital:.2f} (LR={lr:.6f})")
            round_pnl.append(final_capital)
            
        avg_capital = sum(round_pnl) / len(round_pnl)
        print(f"Round {round_idx + 1} Avg Capital: ${avg_capital:.2f}")
        overall_results.append((round_idx + 1, avg_capital, lr))
        
        # Improvement Logic: 
        # If avg capital is low, decrease learning rate and try a "tighter" strategy (handled in C++ via round_idx)
        if avg_capital < 100:
            lr *= 0.7
        else:
            lr *= 1.1 # Reward success with faster learning
            
    with open("iterative_energy_results.txt", "w") as f:
        f.write("Round Avg_Capital LR\n")
        for r in overall_results:
            f.write(f"{r[0]} {r[1]:.4f} {r[2]:.8f}\n")

if __name__ == "__main__":
    run_optimization()
