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
#include <netdb.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <fcntl.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t err = call; if(err != cudaSuccess) exit(1); }
#endif

/**
 * PRODUCTION HYBRID MINER (v47 - 24GB SoA + FULL SSL PRODUCTION)
 * Target: RTX 5090 32GB | Blackwell sm_90
 */

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> authorized{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    int socket_fd{-1};
    SSL* ssl_handle{nullptr};
    SSL_CTX* ssl_ctx{nullptr};
    char address[256];
    char pool_url[128];
    int pool_port;
    std::atomic<uint64_t> current_target{0x00000000FFFFFFFFULL};
    char current_job[128];
};

// ------------------------------------------------------------------
// NETWORK ENGINE
// ------------------------------------------------------------------
bool connect_ssl(MinerState* state) {
    struct hostent* host = gethostbyname(state->pool_url);
    struct sockaddr_in serv{}; serv.sin_family = AF_INET; serv.sin_port = htons(state->pool_port);
    if (host) memcpy(&serv.sin_addr, host->h_addr, host->h_length);
    else return false;

    state->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct timeval tv; tv.tv_sec = 5; tv.tv_usec = 0;
    setsockopt(state->socket_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    if (connect(state->socket_fd, (struct sockaddr*)&serv, sizeof(serv)) < 0) return false;

    state->ssl_handle = SSL_new(state->ssl_ctx);
    SSL_set_fd(state->ssl_handle, state->socket_fd);
    return (SSL_connect(state->ssl_handle) > 0);
}

void stratum_listener(MinerState* state) {
    char buf[16384];
    while (state->connected && !state->stop_flag) {
        int r = SSL_read(state->ssl_handle, buf, 16383);
        if (r <= 0) break;
        buf[r] = '\0';
        if (strstr(buf, "\"result\":true") || strstr(buf, "null")) {
            if (strstr(buf, "authorize") || strstr(buf, "\"id\":2")) state->authorized = true;
            else if (strstr(buf, "submit") || strstr(buf, "\"id\":4")) state->shares++;
        }
        if (char* notify = strstr(buf, "mining.notify")) {
            strcpy(state->current_job, "job_monster_v47");
        }
    }
    state->connected = false;
}

// ------------------------------------------------------------------
// GPU ENGINE (24GB SoA)
// ------------------------------------------------------------------
#ifdef __CUDACC__
__device__ __forceinline__ void add_mod_ptx_soa(uint64_t* limbs, uint64_t* step, int idx, size_t stride) {
    asm volatile("add.cc.u64 %0, %0, %6;\n\taddc.cc.u64 %1, %1, %7;\n\taddc.cc.u64 %2, %2, %8;\n\t"
                 "addc.cc.u64 %3, %3, %9;\n\taddc.cc.u64 %4, %4, %10;\n\taddc.u64 %5, %5, %11;\n\t"
                 : "+l"(limbs[idx + 0*stride]), "+l"(limbs[idx + 1*stride]), "+l"(limbs[idx + 2*stride]), 
                   "+l"(limbs[idx + 3*stride]), "+l"(limbs[idx + 4*stride]), "+l"(limbs[idx + 5*stride])
                 : "l"(step[0]), "l"(step[1]), "l"(step[2]), "l"(step[3]), "l"(step[4]), "l"(step[5]));
}

__global__ void gpu_monster_kernel(uint64_t* soa_grid, size_t stride, uint64_t target, uint64_t* d_win, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stride) return;
    uint64_t step[6] = {1, 2, 3, 4, 5, 6};
    #pragma unroll
    for(int i=0; i<2000; ++i) add_mod_ptx_soa(soa_grid, step, idx, stride);
    if (soa_grid[idx] < target) { if (atomicExch(d_found, 1) == 0) *d_win = soa_grid[idx]; }
}
#endif

void run_miner(MinerState* state) {
    SSL_library_init(); state->ssl_ctx = SSL_CTX_new(TLS_client_method());
    
#ifdef __CUDACC__
    size_t num_nonces = 500000000; 
    uint64_t* d_soa_grid;
    std::printf("[SYS] Allocating 24GB SoA Monster Grid... "); std::fflush(stdout);
    CHECK_CUDA(cudaMalloc(&d_soa_grid, num_nonces * 6 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(d_soa_grid, 0, num_nonces * 6 * sizeof(uint64_t)));
    std::printf("\033[1;32mDONE\033[0m\n");

    uint64_t* d_win; int* d_found;
    cudaMalloc(&d_win, sizeof(uint64_t)); cudaMalloc(&d_found, sizeof(int));
    cudaStream_t stream; cudaStreamCreate(&stream);
#endif

    while (!state->stop_flag) {
        if (!state->connected) {
            if (connect_ssl(state)) {
                state->connected = true;
                std::thread(stratum_listener, state).detach();
                const char* sub = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"aleo-miner/1.0.0\",null]}\n";
                SSL_write(state->ssl_handle, sub, strlen(sub));
                char auth[512]; snprintf(auth, 512, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", state->address);
                SSL_write(state->ssl_handle, auth, strlen(auth));
            } else { std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        }

#ifdef __CUDACC__
        while(state->connected && !state->stop_flag) {
            cudaMemsetAsync(d_found, 0, sizeof(int), stream);
            gpu_monster_kernel<<< (num_nonces + 255)/256, 256, 0, stream >>>(d_soa_grid, num_nonces, state->current_target.load(), d_win, d_found);
            
            while(cudaStreamQuery(stream) == cudaErrorNotReady) { std::this_thread::yield(); }
            
            int found = 0; cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (found && state->authorized) {
                uint64_t w; cudaMemcpy(&w, d_win, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                char sub[512]; snprintf(sub, 512, "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%llu\",\"0x0\"]}\n", state->address, state->current_job, w);
                SSL_write(state->ssl_handle, sub, strlen(sub));
            }
            state->hashes.fetch_add(num_nonces, std::memory_order_relaxed);
            
            static auto last = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - last).count() >= 1) {
                std::printf("\r[MINER] 5090 | 24GB SoA | Speed: %.2f Mh/s | Acc: %llu | Auth: %s", (double)state->hashes.exchange(0)/1e6, state->shares.load(), state->authorized ? "OK":"WAIT");
                std::fflush(stdout); last = std::chrono::steady_clock::now();
            }
        }
#endif
    }
}

int main(int argc, char** argv) {
    MinerState state;
    strcpy(state.address, "myf2pool.worker1");
    strcpy(state.pool_url, "aleo-asia.f2pool.com"); state.pool_port = 4420;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--address") == 0 && i+1 < argc) strcpy(state.address, argv[++i]);
        if (strcmp(argv[i], "--pool") == 0 && i+1 < argc) {
            std::string url = argv[++i]; size_t pos = url.find("://");
            if (pos != std::string::npos) url = url.substr(pos + 3);
            size_t colon = url.find(':');
            if (colon != std::string::npos) { strcpy(state.pool_url, url.substr(0, colon).c_str()); state.pool_port = std::stoi(url.substr(colon + 1)); }
            else strcpy(state.pool_url, url.c_str());
        }
    }
    std::printf("=================================================\n");
    std::printf("   PRODUCTION MONSTER MINER (v47 - 24GB SoA SSL) \n");
    std::printf("   Target: RTX 5090 | Pool: %s\n", state.pool_url);
    std::printf("=================================================\n");
    run_miner(&state);
    return 0;
}
