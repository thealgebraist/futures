#!/bin/bash

SYMBOLS=("SPY" "QQQ" "DIA" "IWM" "VTI" "VXUS" "BND" "VIG")

echo "Running FFNN 256 models (5 mins each)..."
for S in "${SYMBOLS[@]}"; do
    ./ffnn_256 "data/full_history/${S}_max.csv" "$S" >> res_ffnn256.txt
done

echo "Running MC 64 models (5 mins each)..."
for S in "${SYMBOLS[@]}"; do
    ./mc_64 "data/full_history/${S}_max.csv" "$S" >> res_mc64.txt
done

echo "Running GP models (5 mins each)..."
for S in "${SYMBOLS[@]}"; do
    python3 src/indices_gp.py "data/full_history/${S}_max.csv" "$S" >> res_gp.txt
done
