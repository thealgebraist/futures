cmake_minimum_required(VERSION 3.20)
project(AleoOptimizedMiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Dependency Discovery
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
message(STATUS "CURL Found: ${CURL_VERSION}")

# 2. Platform Detection
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_compile_options(-march=native -O3)
elseif(UNIX AND NOT APPLE)
    add_compile_options(-mavx2 -march=native -O3)
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
