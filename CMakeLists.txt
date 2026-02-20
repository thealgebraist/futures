cmake_minimum_required(VERSION 3.20)
project(AleoOptimizedMiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix: Set CUDA standard explicitly to 20 (as 23 is not yet supported by nvcc)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 1. Platform Detection
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(STATUS "Targeting Apple Silicon (NEON Enabled)")
    add_compile_options(-march=native -O3)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Targeting Linux x86_64 (AVX2 Enabled)")
    add_compile_options(-mavx2 -march=native -O3)
endif()

# 2. CUDA Support Detection
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA Detected: Setting up RTX 5090 Targets")
    set(CMAKE_CUDA_ARCHITECTURES 90) # Blackwell sm_90
    
    # Hybrid GPU Miner
    add_executable(hybrid_5090_miner hybrid_5090_miner.cu)
    set_target_properties(hybrid_5090_miner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Standalone CUDA Prover
    add_executable(cuda_prover cuda_prover.cu)
endif()

# 3. CPU Targets
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # Master M2 Prover (ARM Only)
    add_executable(master_m2_prover master_m2_prover.cpp)
    
    # Production Stratum Miner (ARM Version)
    add_executable(production_prover_v2 production_prover_v2.cpp)
    target_link_libraries(production_prover_v2 PRIVATE threads)

    # Real Proof Submitter (ARM Version)
    add_executable(real_proof_submitter real_proof_submitter.cpp)
    target_link_libraries(real_proof_submitter PRIVATE threads)
endif()

# 4. Installation
if(TARGET master_m2_prover)
    install(TARGETS master_m2_prover production_prover_v2 DESTINATION bin)
endif()
