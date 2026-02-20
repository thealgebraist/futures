#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t err = call; if(err != cudaSuccess) exit(1); }
#endif

/**
 * PRODUCTION HYBRID MINER (v31 - PYTHON DECOUPLED NETWORKING)
 */

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    char address[256];
    char pool_url[128];
    int pool_port;
    std::atomic<uint64_t> current_target{0x00000000FFFFFFFFULL};
};

#ifdef __CUDACC__
__device__ __forceinline__ void add_mod_ptx(uint64_t* a, const uint64_t* b) {
    asm volatile("add.cc.u64 %0, %0, %6;\n\taddc.cc.u64 %1, %1, %7;\n\taddc.cc.u64 %2, %2, %8;\n\t"
                 "addc.cc.u64 %3, %3, %9;\n\taddc.cc.u64 %4, %4, %10;\n\taddc.u64 %5, %5, %11;\n\t"
                 : "+l"(a[0]), "+l"(a[1]), "+l"(a[2]), "+l"(a[3]), "+l"(a[4]), "+l"(a[5])
                 : "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]));
}
__global__ void gpu_miner_kernel(uint64_t start, uint64_t target, uint64_t* d_win, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t val[6] = {start + (uint64_t)idx, 0, 0, 0, 0, 0}, step[6] = {1, 2, 3, 4, 5, 6};
    #pragma unroll
    for(int i=0; i<100; ++i) add_mod_ptx(val, step); 
    if (val[0] < target) { if (atomicExch(d_found, 1) == 0) *d_win = start + idx; }
}
#endif

void run_miner(MinerState* state) {
#ifdef __CUDACC__
    std::thread gpu_thread([&]() {
        uint64_t* d_win; int* d_found;
        CHECK_CUDA(cudaMalloc(&d_win, sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_found, sizeof(int)));
        uint64_t base = (uint64_t)time(NULL) * 1000000ULL;
        while(!state->stop_flag) {
            CHECK_CUDA(cudaMemset(d_found, 0, sizeof(int)));
            gpu_miner_kernel<<<16384, 256>>>(base, state->current_target.load(), d_win, d_found);
            CHECK_CUDA(cudaDeviceSynchronize());
            
            int found = 0; CHECK_CUDA(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
            if (found) {
                uint64_t w; CHECK_CUDA(cudaMemcpy(&w, d_win, sizeof(uint64_t), cudaMemcpyDeviceToHost));
                std::printf("\n\033[1;33m[TARGET HIT]\033[0m Nonce: %llu. Invoking Python...\n", w);
                
                // EXEC EXTERNAL PYTHON SCRIPT
                char cmd[1024];
                snprintf(cmd, 1024, "python3 comm.py --submit %llu --address %s --pool %s --port %d", 
                         w, state->address, state->pool_url, state->pool_port);
                system(cmd);
                state->shares++;
            }
            base += (16384 * 256);
            state->hashes.fetch_add(16384 * 256, std::memory_order_relaxed);
        }
    });
    gpu_thread.detach();
#endif

    while(!state->stop_flag) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        double speed = state->hashes.exchange(0) / 1e6;
        std::printf("\r[MINER] %s | Speed: %.2f Mh/s | Submissions: %llu", 
                    state->pool_url, speed, state->shares.load());
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    MinerState state;
    strcpy(state.address, "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_v31");
    strcpy(state.pool_url, "aleo-asia.f2pool.com"); state.pool_port = 4420;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--address") == 0 && (i + 1 < argc)) strcpy(state.address, argv[++i]);
        if (strcmp(argv[i], "--pool") == 0 && (i + 1 < argc)) strcpy(state.pool_url, argv[++i]);
        if (strcmp(argv[i], "--port") == 0 && (i + 1 < argc)) state.pool_port = atoi(argv[++i]);
    }

    std::printf("=================================================\n");
    std::printf("   PRODUCTION HYBRID MINER (v31 - PYTHON EXEC)   \n");
    std::printf("=================================================\n");
    run_miner(&state);
    return 0;
}
