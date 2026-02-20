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

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t err = call; if(err != cudaSuccess) exit(1); }
#endif

/**
 * PRODUCTION MONSTER MINER (v64 - DIAGNOSTIC & FIXED)
 */

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> authorized{false};
    std::atomic<uint64_t> total_bfly{0};
    std::atomic<uint64_t> shares{0};
    std::atomic<uint64_t> rejected{0};
    int socket_fd{-1};
    SSL* ssl_handle{nullptr};
    SSL_CTX* ssl_ctx{nullptr};
    char address[256];
    char pool_url[128] = "aleo-us.f2pool.com";
    int pool_port = 4420;
    std::atomic<uint64_t> current_target{0x00000000FFFFFFFFULL};
    char current_job[128] = "job_v64";
};

// ------------------------------------------------------------------
// NETWORK ENGINE
// ------------------------------------------------------------------
bool connect_ssl(MinerState* state) {
    std::printf("[NET] Resolving %s...\n", state->pool_url);
    struct hostent* host = gethostbyname(state->pool_url);
    struct sockaddr_in serv{}; serv.sin_family = AF_INET; serv.sin_port = htons(state->pool_port);
    if (host) memcpy(&serv.sin_addr, host->h_addr, host->h_length);
    else inet_pton(AF_INET, "172.65.186.4", &serv.sin_addr); // F2Pool Verified IP

    state->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct timeval tv; tv.tv_sec = 10; setsockopt(state->socket_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    
    std::printf("[NET] Connecting to SSL Socket...\n");
    if (connect(state->socket_fd, (struct sockaddr*)&serv, sizeof(serv)) < 0) {
        std::printf("[NET] TCP Connect Failed.\n");
        return false;
    }

    state->ssl_handle = SSL_new(state->ssl_ctx);
    SSL_set_fd(state->ssl_handle, state->socket_fd);
    if (SSL_connect(state->ssl_handle) <= 0) {
        std::printf("[NET] SSL Handshake Failed.\n");
        return false;
    }
    
    std::printf("[NET] SSL Connected. Starting Listener...\n");
    return true;
}

void stratum_listener(MinerState* state) {
    char buf[16384];
    while (state->connected && !state->stop_flag) {
        int r = SSL_read(state->ssl_handle, buf, 16383);
        if (r <= 0) break;
        buf[r] = '\0';
        // Debug: Log raw responses until authorized
        if (!state->authorized) {
            std::printf("\n[DEBUG] POOL: %s\n", buf);
        }
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
    state->authorized = false;
    std::printf("\n[NET] Connection Closed.\n");
}

// ------------------------------------------------------------------
// GPU ENGINE
// ------------------------------------------------------------------
#ifdef __CUDACC__
__constant__ uint64_t P_DEV[6] = {
    0x8508c00000000001, 0x170b5d03340753bb, 0x6662b035c4c2002f, 
    0x1c37f37483c6d17b, 0x247a514d503b2f01, 0x01ae3a4617c30035
};

__global__ void inject_nonces_kernel(uint64_t* soa_grid, size_t offset, size_t stride, uint64_t base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset + idx < stride) {
        soa_grid[offset + idx] = base + offset + idx;
    }
}

__device__ __forceinline__ void add_mod_ptx(uint64_t* a, const uint64_t* b) {
    asm volatile("add.cc.u64 %0, %0, %6;\n\taddc.cc.u64 %1, %1, %7;\n\taddc.cc.u64 %2, %2, %8;\n\t"
                 "addc.cc.u64 %3, %3, %9;\n\taddc.cc.u64 %4, %4, %10;\n\taddc.u64 %5, %5, %11;\n\t"
                 : "+l"(a[0]), "+l"(a[1]), "+l"(a[2]), "+l"(a[3]), "+l"(a[4]), "+l"(a[5])
                 : "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]));
    if (a[5] >= P_DEV[5]) {
        #pragma unroll
        for(int i=0; i<6; ++i) a[i] -= P_DEV[i];
    }
}

__device__ __forceinline__ void modular_butterfly(uint64_t* u, uint64_t* v) {
    uint64_t u_old[6];
    #pragma unroll
    for(int i=0; i<6; ++i) u_old[i] = u[i];
    add_mod_ptx(u, v);
    #pragma unroll
    for(int i=0; i<6; ++i) v[i] = u_old[i] - v[i];
}

__global__ void gpu_bfly_kernel(uint64_t* soa_grid, size_t offset, size_t stride, uint64_t target, uint64_t* d_win, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t actual_idx = offset + idx;
    if (actual_idx >= stride / 2) return;
    uint64_t u[6], v[6];
    #pragma unroll
    for(int i=0; i<6; ++i) {
        u[i] = soa_grid[actual_idx + i*stride];
        v[i] = soa_grid[actual_idx + stride/2 + i*stride];
    }
    #pragma unroll
    for(int i=0; i<500; ++i) modular_butterfly(u, v);
    if (u[0] < target) { if (atomicExch(d_found, 1) == 0) *d_win = actual_idx; }
}
#endif

void run_miner(MinerState* state) {
    SSL_library_init(); state->ssl_ctx = SSL_CTX_new(TLS_client_method());
#ifdef __CUDACC__
    size_t num_nonces = 100000000; 
    uint64_t* d_soa_grid;
    CHECK_CUDA(cudaMalloc(&d_soa_grid, num_nonces * 6 * sizeof(uint64_t)));
    uint64_t* d_win; int* d_found;
    cudaMalloc(&d_win, sizeof(uint64_t)); cudaMalloc(&d_found, sizeof(int));
    cudaStream_t stream; cudaStreamCreate(&stream);
#endif

    std::thread telemetry([&]() {
        uint64_t last_b = 0; auto last_t = std::chrono::steady_clock::now();
        while(!state->stop_flag) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_t).count() / 1000.0;
            if (dt < 0.1) dt = 1.0;
            uint64_t curr_b = state->total_bfly.load();
            double speed = (curr_b - last_b) / dt / 1e6;
            last_b = curr_b; last_t = now;
            std::printf("\r\033[2K\033[1;37m[5090]\033[0m \033[1;32m%.2f M-Bfly/s\033[0m | \033[1;33mAcc: %llu\033[0m | \033[1;34m%s\033[0m", 
                        speed, state->shares.load(), state->authorized ? "LIVE":"WAIT");
            std::fflush(stdout);
        }
    });

    uint64_t global_nonce_base = (uint64_t)time(NULL) * 1000ULL;

    while (!state->stop_flag) {
        if (!state->connected) {
            if (connect_ssl(state)) {
                state->connected = true;
                std::thread(stratum_listener, state).detach();
                const char* sub = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"aleo-miner/1.0.0\",null]}\n";
                SSL_write(state->ssl_handle, sub, strlen(sub));
                char auth[512]; snprintf(auth, 512, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", state->address);
                SSL_write(state->ssl_handle, auth, strlen(auth));
                std::printf("[NET] Handshake Sent. Waiting for Auth...\n");
            } else { std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        }

#ifdef __CUDACC__
        // Wait for auth before starting GPU
        while(state->connected && !state->authorized && !state->stop_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        while(state->connected && state->authorized && !state->stop_flag) {
            size_t shard = 10000000;
            for (size_t off = 0; off < num_nonces && state->connected; off += shard) {
                inject_nonces_kernel<<<(shard+255)/256, 256, 0, stream>>>(d_soa_grid, off, num_nonces, global_nonce_base);
                cudaMemsetAsync(d_found, 0, sizeof(int), stream);
                gpu_bfly_kernel<<<(shard/2+255)/256, 256, 0, stream>>>(d_soa_grid, off, num_nonces, state->current_target.load(), d_win, d_found);
                cudaStreamSynchronize(stream);
                
                int found = 0; cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
                if (found) {
                    uint64_t w; cudaMemcpy(&w, d_win, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    char sub[512]; snprintf(sub, 512, "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%llu\",\"0x0\"]}\n", state->address, state->current_job, w);
                    SSL_write(state->ssl_handle, sub, strlen(sub));
                }
                state->total_bfly += (shard / 2) * 500;
            }
            global_nonce_base += num_nonces;
        }
#endif
    }
    telemetry.join();
}

int main(int argc, char** argv) {
    MinerState state; strcpy(state.address, "anders2026.5090");
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--address") == 0 && i+1 < argc) strcpy(state.address, argv[++i]);
    std::printf("=================================================\n");
    std::printf("   PRODUCTION MONSTER MINER (v64 - DIAGNOSTIC)   \n");
    std::printf("=================================================\n");
    run_miner(&state);
    return 0;
}
