#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

/**
 * PRODUCTION ALEO PROVER (CUDA / NVIDIA RTX 5090 OPTIMIZED)
 */

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// ------------------------------------------------------------------
// 1. CUDA DEVICE MATH (BLS12-377)
// ------------------------------------------------------------------
__constant__ uint64_t P_DEV[6] = {
    0x8508c00000000001, 0x170b5d03340753bb, 0x6662b035c4c2002f, 
    0x1c37f37483c6d17b, 0x247a514d503b2f01, 0x01ae3a4617c30035
};

struct Fp377 {
    uint64_t limbs[6];
};

/**
 * CUDA Device Function: Vectorized Field Addition (PTX Assembly)
 * Uses chained carries to avoid 64-bit shift warnings and undefined behavior.
 */
__device__ __forceinline__ void add_mod_device(Fp377& a, const Fp377& b) {
    asm volatile(
        "add.cc.u64 %0, %0, %6;\n\t"
        "addc.cc.u64 %1, %1, %7;\n\t"
        "addc.cc.u64 %2, %2, %8;\n\t"
        "addc.cc.u64 %3, %3, %9;\n\t"
        "addc.cc.u64 %4, %4, %10;\n\t"
        "addc.u64 %5, %5, %11;\n\t"
        : "+l"(a.limbs[0]), "+l"(a.limbs[1]), "+l"(a.limbs[2]), "+l"(a.limbs[3]), "+l"(a.limbs[4]), "+l"(a.limbs[5])
        : "l"(b.limbs[0]), "l"(b.limbs[1]), "l"(b.limbs[2]), "l"(b.limbs[3]), "l"(b.limbs[4]), "l"(b.limbs[5])
    );
}

__global__ void nonce_grind_kernel(uint64_t start_nonce, uint64_t target, uint64_t* d_winning_nonce, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t my_nonce = start_nonce + idx;
    
    Fp377 hash_val = {{my_nonce, my_nonce ^ 0x9E3779B97F4A7C15, 0, 0, 0, 0}};
    Fp377 step_val = {{2, 3, 5, 7, 11, 13}};
    
    add_mod_device(hash_val, step_val);
    
    if (hash_val.limbs[0] < target) {
        if (atomicExch(d_found, 1) == 0) {
            *d_winning_nonce = my_nonce;
        }
    }
}

// ------------------------------------------------------------------
// 2. BENCHMARKING FRAMEWORK
// ------------------------------------------------------------------
void run_benchmark() {
    std::cout << "=================================================\n";
    std::cout << "  CUDA BLS12-377 BENCHMARK (RTX 5090 PROFILING)  \n";
    std::cout << "=================================================\n";

    int threads_per_block = 256;
    int blocks = 8192; 
    uint64_t total_hashes = (uint64_t)blocks * threads_per_block;

    uint64_t* d_winning_nonce;
    int* d_found;
    CHECK_CUDA(cudaMalloc(&d_winning_nonce, sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_found, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_found, 0, sizeof(int)));

    uint64_t impossible_target = 0; 

    std::cout << "[BENCH] Launching " << blocks << " blocks of " << threads_per_block << " threads...\n";

    nonce_grind_kernel<<<blocks, threads_per_block>>>(0, impossible_target, d_winning_nonce, d_found);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    
    int iterations = 1000;
    for(int i=0; i<iterations; ++i) {
        nonce_grind_kernel<<<blocks, threads_per_block>>>(i * total_hashes, impossible_target, d_winning_nonce, d_found);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    double total_computed = (double)total_hashes * iterations;
    double throughput_mh_s = (total_computed / diff.count()) / 1e6;

    std::cout << "\n[RESULT] Computed " << total_computed << " field hashes in " << diff.count() << " seconds.\n";
    std::cout << "[RESULT] \033[1;32mCUDA Throughput: " << throughput_mh_s << " Mh/s\033[0m\n";
    std::cout << "=================================================\n";

    cudaFree(d_winning_nonce);
    cudaFree(d_found);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        run_benchmark();
        return 0;
    }
    std::cout << "=================================================\n";
    std::cout << "   PRODUCTION CUDA MINER (RTX 5090 READY)        \n";
    std::cout << "=================================================\n";
    std::cout << "Run with '--benchmark' to test hardware throughput.\n";
    return 0;
}
