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
 * PRODUCTION HYBRID MINER (v34 - 14-POOL COMPREHENSIVE AUDIT & RAW DUMP)
 */

struct PoolConfig { const char* name; const char* url; int port; };

PoolConfig AUDIT_LIST[] = {
    {"Apool HK IP", "172.65.162.169", 9090},
    {"F2Pool SSL", "aleo-asia.f2pool.com", 4420},
    {"F2Pool HTTPS", "aleo.f2pool.com", 443},
    {"Oula WSS", "aleo.oula.network", 6666},
    {"ZkWork HK IP", "47.243.163.37", 10003},
    {"ZkWork US", "aleo.us.zk.work", 10003},
    {"AntPool Aleo", "aleo.antpool.com", 9038},
    {"Hpool Global", "aleo.hpool.io", 9090},
    {"AleoPool", "aleo.aleopool.io", 9090},
    {"6Pool Aleo", "aleo.6pool.com", 9090},
    {"ZkWork SG IP", "161.117.82.155", 10003},
    {"Apool US IP", "172.65.230.151", 9090},
    {"WhalePool Asia", "172.65.232.193", 42343},
    {"UniplusPool IP", "217.160.0.235", 9090}
};

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    char address[256];
    char pool_url[128];
    int pool_port;
};

std::string exec_comm(const std::string& cmd) {
    char buffer[256];
    std::string result = "";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "PIPE_ERROR";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) result += buffer;
    pclose(pipe);
    return result;
}

void run_audit(MinerState* state) {
    bool connected = false;
    std::printf("=================================================\n");
    std::printf("   v34 COMPREHENSIVE 14-POOL AUDIT & RAW DUMP    \n");
    std::printf("=================================================\n");

    for (auto& p : AUDIT_LIST) {
        std::printf("\n[AUDIT] %-15s (%s:%d)\n", p.name, p.url, p.port); std::fflush(stdout);
        char cmd[1024];
        snprintf(cmd, 1024, "python3 comm.py --mode test --pool %s --port %d --address %s", p.url, p.port, state->address);
        std::string res = exec_comm(cmd);
        std::printf("%s", res.c_str()); std::fflush(stdout);
        
        if (res.find("OK") != std::string::npos) {
            std::printf("\033[1;32m   >>> SUCCESS: Pool is ONLINE and AUTHORIZED\033[0m\n");
            strcpy(state->pool_url, p.url); state->pool_port = p.port;
            connected = true; break;
        }
    }

    if (!connected) std::printf("\n[FATAL] All 14 targets failed. See DEBUG logs above for blocking reason.\n");
}

int main(int argc, char** argv) {
    MinerState state;
    strcpy(state.address, "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_v34");
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--address") == 0 && i+1 < argc) strcpy(state.address, argv[++i]);
    run_audit(&state);
    return 0;
}
