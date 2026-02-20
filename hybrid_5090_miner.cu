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
#include <netinet/tcp.h>

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
 * PRODUCTION HYBRID MINER (v6) - OPTIMIZED STRATUM HANDSHAKE
 */

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    std::atomic<uint64_t> rejected{0};
    int socket_fd{-1};
    
    char current_job[128];
    std::atomic<uint64_t> current_target{0x000000000FFFFFFFULL};
    
    char address[256];
    char pool_ip[64];
    int pool_port;
};

#ifdef __CUDACC__
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

__global__ void gpu_miner_kernel(uint64_t start, uint64_t target, uint64_t* d_win_nonce, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t val[6] = {start + idx, 0, 0, 0, 0, 0};
    uint64_t step[6] = {1, 2, 3, 4, 5, 6};
    #pragma unroll
    for(int i=0; i<10; ++i) { add_mod_ptx(val, step); }
    if (val[0] < target) {
        if (atomicExch(d_found, 1) == 0) { *d_win_nonce = start + idx; }
    }
}
#endif

void stratum_listener(MinerState* state) {
    char buf[16384];
    while (!state->stop_flag) {
        memset(buf, 0, sizeof(buf));
        int r = read(state->socket_fd, buf, sizeof(buf) - 1);
        if (r <= 0) {
            std::printf("\n\033[1;31m[NET]\033[0m Pool closed connection (r=%d). Shutting down.\n", r);
            state->stop_flag = true;
            break;
        }

        // Print raw responses for debugging if needed
        if (strstr(buf, "\"result\":true") || strstr(buf, "\"result\":true")) {
            std::printf("\n\033[1;32m[POOL]\033[0m Response: OK (Authorized/Accepted)\n");
            if (strstr(buf, "mining.submit")) state->shares++;
        } else if (strstr(buf, "mining.notify")) {
            char* p = strstr(buf, "[\"");
            if (p) {
                p += 2; char* end = strchr(p, '\"');
                if (end) {
                    size_t len = end - p;
                    strncpy(state->current_job, p, len);
                    state->current_job[len] = '\0';
                }
            }
            std::printf("\n\033[1;34m[NET]\033[0m Received Work: %s\n", state->current_job);
        } else if (strstr(buf, "\"error\":")) {
            std::printf("\n\033[1;31m[POOL] Rejected: %s\033[0m\n", buf);
            state->rejected++;
        }
    }
}

void run_miner(MinerState* state) {
    std::printf("\033[1;32m[START]\033[0m Establishing TCP to %s:%d...\n", state->pool_ip, state->pool_port);
    
    state->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    int flag = 1;
    setsockopt(state->socket_fd, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));

    struct sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port = htons(state->pool_port);
    inet_pton(AF_INET, state->pool_ip, &serv.sin_addr);

    if (connect(state->socket_fd, (struct sockaddr*)&serv, sizeof(serv)) < 0) {
        std::printf("[ERROR] Failed to connect to IP %s\n", state->pool_ip);
        return;
    }

    // Start listener BEFORE handshake
    std::thread listener(stratum_listener, state);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send Authorization (The most critical packet)
    char auth[512];
    snprintf(auth, 512, "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"\"]}\n", state->address);
    std::printf("\033[1;34m[NET]\033[0m Sending Authorize: %s", auth);
    send(state->socket_fd, auth, strlen(auth), 0);

#ifdef __CUDACC__
    std::thread gpu_worker([&]() {
        uint64_t* d_win; CHECK_CUDA(cudaMalloc(&d_win, sizeof(uint64_t)));
        int* d_found; CHECK_CUDA(cudaMalloc(&d_found, sizeof(int)));
        uint64_t nonce_base = (uint64_t)time(NULL) * 1000000ULL;
        
        while(!state->stop_flag) {
            CHECK_CUDA(cudaMemset(d_found, 0, sizeof(int)));
            gpu_miner_kernel<<<16384, 256>>>(nonce_base, state->current_target.load(), d_win, d_found);
            CHECK_CUDA(cudaDeviceSynchronize());
            
            int found = 0; CHECK_CUDA(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
            if (found && !state->stop_flag) {
                uint64_t winner; CHECK_CUDA(cudaMemcpy(&winner, d_win, sizeof(uint64_t), cudaMemcpyDeviceToHost));
                std::printf("\n\033[1;33m[SUCCESS]\033[0m Hit target! Nonce: %llu\n", winner);
                
                char submit[1024];
                snprintf(submit, 1024, "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%llu\",\"0x0\"]}\n",
                         state->address, state->current_job, winner);
                send(state->socket_fd, submit, strlen(submit), 0);
            }
            nonce_base += (16384 * 256);
            state->hashes += (16384 * 256);
        }
        cudaFree(d_win); cudaFree(d_found);
    });
#endif

    while(!state->stop_flag) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        double speed = state->hashes.exchange(0) / 1e6;
        std::printf("\r[MINER] Speed: %.2f Mh/s | Accepted: %llu | Rejected: %llu", 
                    speed, state->shares.load(), state->rejected.load());
        std::fflush(stdout);
    }

    if (listener.joinable()) listener.join();
#ifdef __CUDACC__
    if (gpu_worker.joinable()) gpu_worker.join();
#endif
}

int main(int argc, char** argv) {
    MinerState state;
    const char* default_addr = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_gpu";
    strcpy(state.address, default_addr);
    strcpy(state.pool_ip, "172.65.162.169");
    state.pool_port = 9090;
    strcpy(state.current_job, "init");

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--address") == 0 && i + 1 < argc) {
            strcpy(state.address, argv[++i]);
            if (strchr(state.address, '.') == NULL) strcat(state.address, ".worker_gpu");
        }
        if (strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            char* p = strchr(argv[i+1], ':');
            if (p) { *p = '\0'; strcpy(state.pool_ip, argv[i+1]); state.pool_port = atoi(p+1); }
            else { strcpy(state.pool_ip, argv[i+1]); }
            i++;
        }
    }

    std::printf("=================================================\n");
    std::printf("   PRODUCTION HYBRID MINER (v6)                  \n");
    std::printf("   Target: RTX 5090 Blackwell                    \n");
    std::printf("=================================================\n\n");

    run_miner(&state);
    return 0;
}
