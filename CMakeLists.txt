cmake_minimum_required(VERSION 3.20)
project(AleoOptimizedMiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Dependency Discovery (Ubuntu Standard Paths)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

# 2. Linux Optimization Flags
if(UNIX AND NOT APPLE)
    message(STATUS "Targeting Linux x86_64 (AVX2 + sm_90 Enabled)")
    add_compile_options(-mavx2 -march=native -O3 -pthread)
endif()

# 3. CUDA Support Detection
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 90) # Blackwell sm_90
    set(CMAKE_CUDA_STANDARD 20)
    
    add_executable(hybrid_5090_miner hybrid_5090_miner.cu)
    target_link_libraries(hybrid_5090_miner PRIVATE 
        CURL::libcurl 
        OpenSSL::SSL 
        OpenSSL::Crypto 
        Threads::Threads)
endif()
