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
#include <sqlite3.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { cudaError_t err = call; if(err != cudaSuccess) { std::printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } }
#endif

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> authorized{false};
    std::atomic<uint64_t> total_bfly{0};
    std::atomic<uint64_t> shares{0};
    char address[256];
    char pool_url[128] = "aleo-us.f2pool.com";
    int pool_port = 4420;
    std::atomic<uint64_t> current_target{0x0000000000000FFFULL};
    char current_job[128] = "job_v69";
    sqlite3* db{nullptr};
    SSL* ssl_handle{nullptr};
    SSL_CTX* ssl_ctx{nullptr};
    int socket_fd{-1};
};

// ------------------------------------------------------------------
// PERSISTENCE & NETWORK
// ------------------------------------------------------------------
void init_db(MinerState* state) {
    sqlite3_open("shares_cache.db", &state->db);
    sqlite3_exec(state->db, "CREATE TABLE IF NOT EXISTS shares (job_id TEXT, nonce UNSIGNED BIG INT);", 0, 0, 0);
}

bool connect_ssl(MinerState* state) {
    struct hostent* host = gethostbyname(state->pool_url);
    struct sockaddr_in serv{}; serv.sin_family = AF_INET; serv.sin_port = htons(state->pool_port);
    if (host) memcpy(&serv.sin_addr, host->h_addr, host->h_length);
    else inet_pton(AF_INET, "172.65.186.4", &serv.sin_addr);
    state->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct timeval tv; tv.tv_sec = 10; setsockopt(state->socket_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
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
            char* p = strstr(notify, "[\"");
            if (p) { p+=2; char* e = strchr(p, '\"'); if(e){ strncpy(state->current_job, p, e-p); state->current_job[e-p] = '\0'; } }
        }
    }
    state->connected = false; state->authorized = false;
}

// ------------------------------------------------------------------
// COMPUTE KERNELS
// ------------------------------------------------------------------
#ifdef __CUDACC__
__constant__ uint64_t P_DEV[6] = {0x8508c00000000001, 0x170b5d03340753bb, 0x6662b035c4c2002f, 0x1c37f37483c6d17b, 0x247a514d503b2f01, 0x01ae3a4617c30035};
__device__ __forceinline__ void add_mod_ptx(uint64_t* a, const uint64_t* b) {
    asm volatile("add.cc.u64 %0, %0, %6;\n\taddc.cc.u64 %1, %1, %7;\n\taddc.cc.u64 %2, %2, %8;\n\taddc.cc.u64 %3, %3, %9;\n\taddc.cc.u64 %4, %4, %10;\n\taddc.u64 %5, %5, %11;\n\t"
                 : "+l"(a[0]), "+l"(a[1]), "+l"(a[2]), "+l"(a[3]), "+l"(a[4]), "+l"(a[5]) : "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]));
    if (a[5] >= P_DEV[5]) { #pragma unroll
        for(int i=0; i<6; ++i) a[i] -= P_DEV[i]; }
}
__global__ void gpu_bfly_kernel(uint64_t* soa, size_t off, size_t stride, uint64_t target, uint64_t* d_win, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; size_t aidx = off + idx; if (aidx >= stride/2) return;
    uint64_t u[6], v[6]; for(int i=0; i<6; ++i){ u[i] = soa[aidx + i*stride]; v[i] = soa[aidx + stride/2 + i*stride]; }
    for(int i=0; i<500; ++i){ uint64_t uo[6]; for(int j=0; j<6; ++j) uo[j]=u[j]; add_mod_ptx(u, v); for(int j=0; j<6; ++j) v[j]=uo[j]-v[j]; }
    if (u[0] < target) { if(atomicExch(d_found, 1) == 0) *d_win = aidx; }
}
#endif

void run_miner(MinerState* state) {
    SSL_library_init(); state->ssl_ctx = SSL_CTX_new(TLS_client_method()); init_db(state);
    std::thread telemetry([&]() {
        uint64_t lb = 0; auto lt = std::chrono::steady_clock::now();
        while(!state->stop_flag) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lt).count() / 1000.0;
            uint64_t cb = state->total_bfly.load();
            std::printf("\r\033[2K\033[1;37m[MINER]\033[0m \033[1;32m%.2f M-Bfly/s\033[0m | \033[1;33mAcc: %llu\033[0m | \033[1;34m%s\033[0m", (double)(cb-lb)/dt/1e6, state->shares.load(), state->authorized ? "LIVE":"WAIT");
            lb = cb; lt = now; std::fflush(stdout);
        }
    });

    while (!state->stop_flag) {
        if (!state->connected) {
            if (connect_ssl(state)) {
                state->connected = true; std::thread(stratum_listener, state).detach();
                char auth[512]; snprintf(auth, 512, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", state->address);
                SSL_write(state->ssl_handle, auth, strlen(auth));
            } else std::this_thread::sleep_for(std::chrono::seconds(5));
        }
#ifdef __CUDACC__
        size_t n = 100000000; uint64_t *dsoa, *dw; int *df; 
        cudaMalloc(&dsoa, n*6*8); cudaMemset(dsoa, 1, n*6*8);
        cudaMalloc(&dw, 8); cudaMalloc(&df, 4); cudaStream_t s; cudaStreamCreate(&s);
        while(state->connected && state->authorized) {
            cudaMemsetAsync(df, 0, 4, s);
            gpu_bfly_kernel<<<(10000000+255)/256, 256, 0, s>>>(dsoa, 0, n, state->current_target.load(), dw, df);
            cudaStreamSynchronize(s); int f=0; cudaMemcpy(&f, df, 4, cudaMemcpyDeviceToHost);
            if(f) state->shares++; state->total_bfly += 5000000000ULL;
        }
#endif
    }
}

int main(int argc, char** argv) {
    MinerState state; strcpy(state.address, "anders2026.5090");
    run_miner(&state); return 0;
}
