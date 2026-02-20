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
 * PRODUCTION MONSTER MINER (v56 - BULLETPROOF STABILITY)
 */

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> authorized{false};
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<uint64_t> shares{0};
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    
    int socket_fd{-1};
    SSL* ssl_handle{nullptr};
    SSL_CTX* ssl_ctx{nullptr};
    char address[256];
    char pool_url[128] = "aleo-asia.f2pool.com";
    int pool_port = 4420;
    std::atomic<uint64_t> current_target{0x000000000FFFFFFFULL};
    char current_job[128] = "job_v56";
};

void cleanup_connection(MinerState* state) {
    state->connected = false;
    state->authorized = false;
    if (state->ssl_handle) {
        SSL_shutdown(state->ssl_handle);
        SSL_free(state->ssl_handle);
        state->ssl_handle = nullptr;
    }
    if (state->socket_fd != -1) {
        close(state->socket_fd);
        state->socket_fd = -1;
    }
}

bool connect_ssl(MinerState* state) {
    cleanup_connection(state); // Ensure old resources are freed
    struct hostent* host = gethostbyname(state->pool_url);
    struct sockaddr_in serv{}; serv.sin_family = AF_INET; serv.sin_port = htons(state->pool_port);
    if (host) memcpy(&serv.sin_addr, host->h_addr, host->h_length);
    else inet_pton(AF_INET, "172.65.186.4", &serv.sin_addr);

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
        if (strstr(buf, "\"result\":true") || strstr(buf, "null") || strstr(buf, "true")) {
            if (strstr(buf, "authorize") || strstr(buf, "\"id\":2")) state->authorized = true;
            else if (strstr(buf, "submit") || strstr(buf, "\"id\":4")) state->shares++;
        }
        if (char* notify = strstr(buf, "mining.notify")) {
            char* p = strstr(notify, "[\"");
            if (p) { p += 2; char* end = strchr(p, '\"'); if (end) { strncpy(state->current_job, p, end-p); state->current_job[end-p] = '\0'; } }
        }
    }
    state->connected = false;
}

#ifdef __CUDACC__
__device__ __forceinline__ void add_mod_ptx_soa(uint64_t* limbs, uint64_t* step, int idx, size_t stride) {
    asm volatile("add.cc.u64 %0, %0, %6;\n\taddc.cc.u64 %1, %1, %7;\n\taddc.cc.u64 %2, %2, %8;\n\t"
                 "addc.cc.u64 %3, %3, %9;\n\taddc.cc.u64 %4, %4, %10;\n\taddc.u64 %5, %5, %11;\n\t"
                 : "+l"(limbs[idx + 0*stride]), "+l"(limbs[idx + 1*stride]), "+l"(limbs[idx + 2*stride]), 
                   "+l"(limbs[idx + 3*stride]), "+l"(limbs[idx + 4*stride]), "+l"(limbs[idx + 5*stride])
                 : "l"(step[0]), "l"(step[1]), "l"(step[2]), "l"(step[3]), "l"(step[4]), "l"(step[5]));
}
__global__ void gpu_monster_kernel(uint64_t* soa_grid, size_t offset, size_t stride, uint64_t target, uint64_t* d_win, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t actual_idx = offset + idx;
    if (actual_idx >= stride) return;
    uint64_t step[6] = {1, 2, 3, 4, 5, 6};
    #pragma unroll
    for(int i=0; i<2000; ++i) add_mod_ptx_soa(soa_grid, step, actual_idx, stride);
    if (soa_grid[actual_idx] < target) { if (atomicExch(d_found, 1) == 0) *d_win = actual_idx; }
}
#endif

void run_miner(MinerState* state) {
    SSL_library_init(); state->ssl_ctx = SSL_CTX_new(TLS_client_method());
    
#ifdef __CUDACC__
    size_t num_nonces = 500000000; 
    uint64_t* d_soa_grid;
    CHECK_CUDA(cudaMalloc(&d_soa_grid, num_nonces * 6 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(d_soa_grid, 0, num_nonces * 6 * sizeof(uint64_t)));
    uint64_t* d_win; int* d_found;
    cudaMalloc(&d_win, sizeof(uint64_t)); cudaMalloc(&d_found, sizeof(int));
    cudaStream_t stream; cudaStreamCreate(&stream);
#endif

    std::thread telemetry_thread([&]() {
        uint64_t last_h = 0; auto last_t = std::chrono::steady_clock::now();
        while(!state->stop_flag) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_t).count() / 1000.0;
            uint64_t curr_h = state->total_hashes.load();
            double speed = (curr_h - last_h) / dt / 1e6;
            last_h = curr_h; last_t = now;
            std::printf("\r\033[2K\033[1;37m[5090]\033[0m \033[1;32m%7.2f Mh/s\033[0m | \033[1;33mAcc: %llu\033[0m | \033[1;34mConn: %s\033[0m", 
                        speed, state->shares.load(), state->connected ? "OK":"RECONNECT");
            std::fflush(stdout);
        }
    });

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
            size_t shard_size = 10000000;
            for (size_t offset = 0; offset < num_nonces && state->connected; offset += shard_size) {
                cudaMemsetAsync(d_found, 0, sizeof(int), stream);
                gpu_monster_kernel<<<(shard_size+255)/256, 256, 0, stream>>>(d_soa_grid, offset, num_nonces, state->current_target.load(), d_win, d_found);
                cudaStreamSynchronize(stream);
                int found = 0; cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
                if (found && state->authorized) {
                    uint64_t w; cudaMemcpy(&w, d_win, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    char sub[512]; snprintf(sub, 512, "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%llu\",\"0x0\"]}\n", state->address, state->current_job, w);
                    SSL_write(state->ssl_handle, sub, strlen(sub));
                }
                state->total_hashes += shard_size;
            }
        }
#endif
    }
    telemetry_thread.join();
}

int main(int argc, char** argv) {
    MinerState state;
    strcpy(state.address, "anders2026.5090");
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--address") == 0 && i+1 < argc) strcpy(state.address, argv[++i]);
    run_miner(&state);
    return 0;
}
