#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cstdio>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/**
 * PRODUCTION HYBRID MINER (v35 - SNI MASKING & DIRECT IP)
 */

struct PoolConfig { const char* name; const char* ip; int port; };

PoolConfig AUDIT_LIST[] = {
    {"Google DNS", "8.8.8.8", 53},
    {"F2Pool HK SSL", "172.65.186.4", 443},
    {"Apool HK", "172.65.162.169", 9090},
    {"Apool US", "172.65.230.151", 9090},
    {"ZkWork HK", "47.243.163.37", 10003},
    {"ZkWork US", "172.65.230.151", 10003},
    {"WhalePool Asia", "172.65.232.193", 42343},
    {"Oula Asia", "47.237.70.148", 6666},
    {"UniplusPool", "217.160.0.235", 9090},
    {"Hpool Global", "119.28.140.245", 9090}
};

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    char address[256];
    char pool_ip[128];
    int pool_port;
};

std::string exec_comm(const std::string& cmd) {
    char buffer[256];
    std::string result = "";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "PIPE_ERR";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) result += buffer;
    pclose(pipe);
    return result;
}

void run_audit(MinerState* state) {
    bool connected = false;
    std::printf("=================================================\n");
    std::printf("   v35 STEALTH AUDIT: SNI MASKING & DIRECT IP    \n");
    std::printf("=================================================\n");

    for (auto& p : AUDIT_LIST) {
        std::printf("[AUDIT] %-15s (%s:%d) ... ", p.name, p.ip, p.port); std::fflush(stdout);
        char cmd[1024];
        snprintf(cmd, 1024, "python3 comm.py --mode test --pool %s --port %d --address %s", p.ip, p.port, state->address);
        std::string res = exec_comm(cmd);
        
        if (res.find("OK") != std::string::npos || res.find("INTERNET_OK") != std::string::npos) {
            std::printf("\033[1;32m%s\033[0m\n", res.c_str());
            if (strcmp(p.ip, "8.8.8.8") != 0) {
                strcpy(state->pool_ip, p.ip); state->pool_port = p.port;
                connected = true; break;
            }
        } else {
            std::printf("\033[1;31m%s\033[0m", res.c_str());
        }
    }

    if (!connected) std::printf("\n[FATAL] No open routes found. Outbound traffic is heavily restricted.\n");
}

int main(int argc, char** argv) {
    MinerState state;
    strcpy(state.address, "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_v35");
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--address") == 0 && i+1 < argc) strcpy(state.address, argv[++i]);
    run_audit(&state);
    return 0;
}
