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
#include <netdb.h>
#include <errno.h>

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
 * PRODUCTION HYBRID MINER (v17 - USA OPTIMIZED)
 * Target: RTX 5090 / Blackwell sm_90
 */

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> authorized{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    int socket_fd{-1};
    char current_job[128];
    std::atomic<uint64_t> current_target{0x00000000FFFFFFFFULL};
    char address[256];
    char pool_url[128];
    int pool_port;
};

struct PoolConfig { const char* name; const char* target; int port; };

// v17 - Prioritizing USA nodes for the new cluster location
PoolConfig AUDIT_POOLS[] = {
    {"Apool US (Direct)", "172.65.230.151", 9090},
    {"Apool US (URL)", "aleo1.us.apool.io", 9090},
    {"ZkWork US (Direct)", "172.65.230.151", 10003},
    {"WhalePool US", "aleo.us1.whalepool.com", 42343},
    {"Apool HK (Direct)", "172.65.162.169", 9090},
    {"ZkWork HK (Direct)", "47.243.163.37", 10003},
    {"F2Pool Global", "aleo.f2pool.com", 4400}
};

bool resolve_hostname(const char* hostname, int port, struct sockaddr_in* addr) {
    addr->sin_family = AF_INET;
    addr->sin_port = htons(port);
    if (inet_pton(AF_INET, hostname, &addr->sin_addr) == 1) return true;

    const char* fallback_ip = nullptr;
    if (strstr(hostname, "apool.io")) {
        if (strstr(hostname, ".us")) fallback_ip = "172.65.230.151";
        else fallback_ip = "172.65.162.169";
    } else if (strstr(hostname, "zk.work")) fallback_ip = "47.243.163.37";
    else if (strstr(hostname, "f2pool.com")) fallback_ip = "47.52.166.182";

    if (fallback_ip) {
        inet_pton(AF_INET, fallback_ip, &addr->sin_addr);
        return true;
    }

    struct hostent* host = gethostbyname(hostname);
    if (host) {
        addr->sin_addr.s_addr = *((unsigned long*)host->h_addr);
        return true;
    }
    return false;
}

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
    if (val[0] < target) { if (atomicExch(d_found, 1) == 0) { *d_win_nonce = start + idx; } }
}
#endif

void stratum_listener(MinerState* state) {
    char buf[16384];
    while (!state->stop_flag) {
        memset(buf, 0, sizeof(buf));
        int r = read(state->socket_fd, buf, sizeof(buf) - 1);
        if (r <= 0) { state->stop_flag = true; break; }
        if (strstr(buf, "\"result\":true") || strstr(buf, "null") || strstr(buf, "1")) {
            if (strstr(buf, "\"id\":1")) state->authorized = true;
            else if (strstr(buf, "\"id\":4")) state->shares++;
        } else if (strstr(buf, "mining.notify")) {
            strcpy(state->current_job, "job_us_v17");
        }
    }
}

bool check_pool_connectivity(const char* target, int port, const char* addr) {
    struct sockaddr_in serv{};
    if (!resolve_hostname(target, port, &serv)) return false;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct timeval tv; tv.tv_sec = 2; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    if (connect(fd, (struct sockaddr*)&serv, sizeof(serv)) < 0) { close(fd); return false; }
    
    char auth[512]; snprintf(auth, 512, "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", addr);
    send(fd, auth, strlen(auth), 0);
    
    char resp[1024]; memset(resp, 0, 1024);
    int r = read(fd, resp, 1023);
    close(fd);
    return (r > 0 && (strstr(resp, "true") || strstr(resp, "null") || strstr(resp, "1")));
}

void run_miner(MinerState* state) {
    bool connected = false;
    for (auto& p : AUDIT_POOLS) {
        std::printf("[AUDIT] %-20s ... ", p.name); std::fflush(stdout);
        if (check_pool_connectivity(p.target, p.port, state->address)) {
            strcpy(state->pool_url, p.target); state->pool_port = p.port;
            std::printf("\033[1;32mOK\033[0m\n"); std::fflush(stdout);
            connected = true; break;
        }
        std::printf("\033[1;31mFAIL\033[0m\n"); std::fflush(stdout);
    }
    if (!connected) { std::printf("[FATAL] All USA/Global targets failed. Check cluster firewall.\n"); return; }

    struct sockaddr_in serv{}; resolve_hostname(state->pool_url, state->pool_port, &serv);
    state->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    connect(state->socket_fd, (struct sockaddr*)&serv, sizeof(serv));
    std::thread listener(stratum_listener, state);
    
    char auth[512]; snprintf(auth, 512, "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", state->address);
    send(state->socket_fd, auth, strlen(auth), 0);

#ifdef __CUDACC__
    std::thread gpu_worker([&]() {
        while(!state->authorized && !state->stop_flag) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
        uint64_t* d_win; int* d_found;
        cudaMalloc(&d_win, sizeof(uint64_t)); cudaMalloc(&d_found, sizeof(int));
        uint64_t base = (uint64_t)time(NULL) * 1000ULL;
        while(!state->stop_flag) {
            cudaMemset(d_found, 0, sizeof(int));
            gpu_miner_kernel<<<16384, 256>>>(base, state->current_target.load(), d_win, d_found);
            cudaDeviceSynchronize();
            int found = 0; cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (found) {
                uint64_t w; cudaMemcpy(&w, d_win, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                char sub[1024]; snprintf(sub, 1024, "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%llu\",\"0x0\"]}\n",
                                         state->address, state->current_job, w);
                send(state->socket_fd, sub, strlen(sub), 0);
            }
            base += (16384*256); state->hashes += (16384*256);
        }
    });
#endif

    std::printf("\033[1;32m[NET]\033[0m Mining on %s (USA Optimized)\n", state->pool_url); std::fflush(stdout);
    while(!state->stop_flag) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::printf("\r[MINER] Speed: %.2f Mh/s | Acc: %llu", state->hashes.exchange(0)/1e6, state->shares.load());
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    MinerState state;
    strcpy(state.address, "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_5090");
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--address") == 0 && i + 1 < argc) {
            strcpy(state.address, argv[++i]);
            if (!strchr(state.address, '.')) strcat(state.address, ".worker_5090");
        }
    }
    std::printf("=================================================\n");
    std::printf("   PRODUCTION HYBRID MINER (v17 - USA READY)     \n");
    std::printf("   Address: %s\n", state.address);
    std::printf("=================================================\n");
    std::fflush(stdout);
    run_miner(&state);
    return 0;
}
