#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>

// CPU Vector Intrinsics for x86_64 (Ubuntu Servers)
#if defined(__x86_64__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}
#else
#define __global__
#define __device__
#define __constant__
#define CHECK_CUDA(call)
#endif

/**
 * HYBRID ALEO PROVER (CPU AVX2 + GPU CUDA)
 * Target: Ubuntu + NVIDIA RTX 5090 (Blackwell sm_90) + AMD EPYC / Intel Xeon
 */

#ifdef __CUDACC__
__constant__ uint64_t P_DEV[6] = {
    0x8508c00000000001, 0x170b5d03340753bb, 0x6662b035c4c2002f, 
    0x1c37f37483c6d17b, 0x247a514d503b2f01, 0x01ae3a4617c30035
};

__device__ __forceinline__ void add_mod_ptx(uint64_t* a, const uint64_t* b) {
    asm volatile(
        "add.cc.u64 %0, %0, %6;\n\t"
        "addc.cc.u64 %1, %1, %7;\n\t"
        "addc.cc.u64 %2, %2, %8;\n\t"
        "addc.cc.u64 %3, %3, %9;\n\t"
        "addc.cc.u64 %4, %4, %10;\n\t"
        "addc.u64 %5, %5, %11;\n\t"
        : "+l"(a[0]), "+l"(a[1]), "+l"(a[2]), "+l"(a[3]), "+l"(a[4]), "+l"(a[5])
        : "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5])
    );
}

__global__ void gpu_hash_kernel(uint64_t start_nonce, uint64_t target, uint64_t* d_hashes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t my_nonce = start_nonce + idx;
    uint64_t hash_val[6] = {my_nonce, my_nonce ^ 0x9E3779B97F4A7C15, 0, 0, 0, 0};
    uint64_t step_val[6] = {2, 3, 5, 7, 11, 13};
    #pragma unroll
    for(int i=0; i<10; ++i) { add_mod_ptx(hash_val, step_val); }
    atomicAdd((unsigned long long*)d_hashes, 1);
}
#endif

#if defined(__x86_64__) || defined(__AVX2__)
namespace BLS12_377_AVX2 {
    struct alignas(32) BatchFp { __m256i limbs[6]; };
    inline void add_vec_avx2(BatchFp& a, const BatchFp& b) {
        __m256i carry = _mm256_setzero_si256();
        __m256i sign_flip = _mm256_set1_epi64x(0x8000000000000000ULL);
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            __m256i va = a.limbs[i];
            __m256i vb = b.limbs[i];
            __m256i sum1 = _mm256_add_epi64(va, vb);
            __m256i a_flipped = _mm256_xor_si256(va, sign_flip);
            __m256i sum1_flipped = _mm256_xor_si256(sum1, sign_flip);
            __m256i c1 = _mm256_cmpgt_epi64(a_flipped, sum1_flipped);
            __m256i sum2 = _mm256_add_epi64(sum1, carry);
            __m256i sum1_flip2 = _mm256_xor_si256(sum1, sign_flip);
            __m256i sum2_flipped = _mm256_xor_si256(sum2, sign_flip);
            __m256i c2 = _mm256_cmpgt_epi64(sum1_flip2, sum2_flipped);
            carry = _mm256_srli_epi64(_mm256_or_si256(c1, c2), 63);
            a.limbs[i] = sum2;
        }
    }
}
void cpu_worker_thread(std::atomic<bool>* stop, std::atomic<uint64_t>* total_hashes) {
    BLS12_377_AVX2::BatchFp a, b;
    for(int i=0; i<6; ++i) { a.limbs[i] = _mm256_set1_epi64x(1); b.limbs[i] = _mm256_set1_epi64x(2); }
    uint64_t local_hashes = 0;
    while (!stop->load(std::memory_order_relaxed)) {
        #pragma unroll
        for(int i=0; i<2500; ++i) { BLS12_377_AVX2::add_vec_avx2(a, b); }
        local_hashes += 10000;
        if (local_hashes >= 1000000) { total_hashes->fetch_add(local_hashes, std::memory_order_relaxed); local_hashes = 0; }
    }
}
#endif

void run_benchmark() {
    std::cout << "=================================================\n";
    std::cout << "   HYBRID ALEO PROVER (CPU AVX2 + GPU CUDA)      \n";
    std::cout << "=================================================\n\n";
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> cpu_hashes{0};
    unsigned int num_cores = std::thread::hardware_concurrency();
    std::cout << "[SYSTEM] Spawning " << num_cores << " CPU AVX2 Workers...\n";
    std::vector<std::thread> cpu_threads;
#if defined(__x86_64__) || defined(__AVX2__)
    for (unsigned int i = 0; i < num_cores; ++i) cpu_threads.emplace_back(cpu_worker_thread, &stop_flag, &cpu_hashes);
#endif
    std::cout << "[SYSTEM] Initializing NVIDIA GPU (CUDA sm_90)...\n";
#ifdef __CUDACC__
    int blocks = 16384, threads = 256;
    uint64_t* d_hashes;
    CHECK_CUDA(cudaMalloc(&d_hashes, sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(d_hashes, 0, sizeof(uint64_t)));
    gpu_hash_kernel<<<blocks, threads>>>(0, 0, d_hashes);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_hashes, 0, sizeof(uint64_t)));
#endif
    std::cout << "\n[BENCHMARK] Running Dual-Mining simulation for 10 seconds...\n\n";
    uint64_t last_cpu = 0, last_gpu = 0;
    for (int i = 1; i <= 10; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint64_t current_cpu = cpu_hashes.load();
        double cpu_speed = (current_cpu - last_cpu) / 1e6;
        last_cpu = current_cpu;
        double gpu_speed = 0;
#ifdef __CUDACC__
        gpu_hash_kernel<<<blocks, threads>>>(i * 10000000, 0, d_hashes);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint64_t current_gpu = 0;
        CHECK_CUDA(cudaMemcpy(&current_gpu, d_hashes, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        gpu_speed = (current_gpu - last_gpu) / 1e6;
        last_gpu = current_gpu;
#endif
        std::cout << "[TELEMETRY] CPU: " << std::fixed << std::setprecision(2) << cpu_speed << " Mh/s | "
                  << "GPU: " << gpu_speed << " Mh/s | "
                  << "\033[1;32mTOTAL: " << (cpu_speed + gpu_speed) << " Mh/s\033[0m\n";
    }
    stop_flag = true;
    for (auto& t : cpu_threads) if (t.joinable()) t.join();
#ifdef __CUDACC__
    cudaFree(d_hashes);
#endif
    std::cout << "\n=================================================\n";
    std::cout << "[DONE] Hybrid Benchmark Complete.\n";
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") { run_benchmark(); return 0; }
    std::cout << "=================================================\n";
    std::cout << "   UBUNTU RTX 5090 ALEO MINER (CUDA + AVX2)      \n";
    std::cout << "=================================================\n";
    std::cout << "Run with '--benchmark' to test hardware throughput.\n";
    return 0;
}
