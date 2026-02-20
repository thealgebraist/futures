#!/bin/bash
# Aleo Miner Ubuntu Setup Script
echo "[SYS] Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y 
    build-essential 
    cmake 
    libcurl4-openssl-dev 
    libssl-dev 
    pkg-config 
    python3-pip

echo "[SYS] Setup complete. Ready to compile with NVCC."
