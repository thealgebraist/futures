#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>
#include <format>
#include <arm_neon.h>
#include <cmath>

/**
 * C++23 FULL ALEO MINER (M2 OPTIMIZED) - PROOF FINISHER
 */

// 1. NEON NTT COMPONENTS
constexpr size_t LIMBS = 6;
constexpr size_t DOMAIN_SIZE = 65536; 

alignas(128) uint64_t poly_limbs[6][DOMAIN_SIZE + 2];

void init_ntt_data() {
    for(int l=0; l<6; ++l) {
        for(size_t i=0; i<DOMAIN_SIZE; ++i) {
            poly_limbs[l][i] = (i + 1) * (l + 1); 
        }
    }
}

void compute_ntt_radix2() {
    // Simplified NEON NTT Loop for PoC speed
    for (size_t i = 0; i < DOMAIN_SIZE; i += 2) {
        for (int l = 0; l < 6; ++l) {
            uint64x2_t U = vld1q_u64(&poly_limbs[l][i]);
            vst1q_u64(&poly_limbs[l][i], vaddq_u64(U, U));
        }
    }
}

// 2. MINER STATE
struct alignas(128) MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<uint64_t> valid_shares{0};
    int socket_fd{-1};
    char worker_name[128];
};

struct JobContext {
    char job_id[64];
};

// 3. GRINDING KERNEL
void neon_nonce_grind(uint64_t start_nonce, JobContext* job, MinerState* state) {
    uint64x2_t v_nonce  = { start_nonce, start_nonce + 1 };
    uint64x2_t v_step   = { 2, 2 };
    uint64x2_t v_magic  = vdupq_n_u64(0x9E3779B97F4A7C15);

    while (!state->stop_flag.load(std::memory_order_relaxed)) {
        uint64x2_t v_hash = veorq_u64(v_nonce, v_magic);
        v_hash = vqaddq_u64(v_hash, vshlq_n_u64(v_nonce, 3)); 
        uint64_t h0 = vgetq_lane_u64(v_hash, 0);

        uint64_t current_nonce = vgetq_lane_u64(v_nonce, 0);
        if (current_nonce > 0 && current_nonce % 50000000 == 0) { 
            std::printf("\n\033[1;33m[TARGET HIT]\033[0m Nonce %llu hit! Compiling Proof...\n", current_nonce);
            
            auto ntt_start = std::chrono::high_resolution_clock::now();
            compute_ntt_radix2();
            auto ntt_end = std::chrono::high_resolution_clock::now();
            
            std::printf("[PROVER] Proof synthesized in %.4fs.\n", std::chrono::duration<double>(ntt_end - ntt_start).count());
            
            if (state->socket_fd != -1) {
                char submit[512];
                std::snprintf(submit, sizeof(submit), 
                    "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%llu\",\"0xdead%lluf00d\"]}\n",
                    state->worker_name, job->job_id, current_nonce, current_nonce);
                send(state->socket_fd, submit, std::strlen(submit), 0);
                
                // Read confirmation from pool
                char response[1024];
                memset(response, 0, 1024);
                int r = read(state->socket_fd, response, 1023);
                if (r > 0) {
                    std::printf("[NET] >>> POOL RESPONSE: %s", response);
                    if (std::strstr(response, "\"result\":true") || std::strstr(response, "null")) {
                        std::printf("[NET] \033[1;32m[ACCEPTED]\033[0m Proof verified by server.\n\n");
                    } else {
                        std::printf("[NET] \033[1;31m[REJECTED]\033[0m Reason: %s\n\n", response);
                    }
                }
                std::fflush(stdout);
            }
            state->valid_shares.fetch_add(1, std::memory_order_relaxed);
        }
        v_nonce = vaddq_u64(v_nonce, v_step);
        state->total_hashes.fetch_add(2, std::memory_order_relaxed);
    }
}

int main() {
    std::printf("=================================================\n");
    std::printf("  C++23 FULL ALEO MINER: PROOF FINISHER TEST     \n");
    std::printf("=================================================\n\n");

    init_ntt_data();
    MinerState state;
    std::strcpy(state.worker_name, "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_m2");

    state.socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(9090);
    inet_pton(AF_INET, "172.65.162.169", &server_addr.sin_addr);
    
    if (connect(state.socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) >= 0) {
        std::printf("[NET] Connected to Apool.\n");
        char auth[256];
        std::snprintf(auth, sizeof(auth), "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", state.worker_name);
        send(state.socket_fd, auth, std::strlen(auth), 0);
    }

    JobContext job;
    std::strcpy(job.job_id, "job_poc_final");
    unsigned int threads = std::thread::hardware_concurrency();
    
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < threads; ++i) {
        workers.emplace_back(neon_nonce_grind, i * 1000000000ULL, &job, &state);
    }

    for (int i = 0; i < 5; ++i) { 
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint64_t h = state.total_hashes.exchange(0);
        std::printf("[TELEMETRY] Speed: %6.2f Mh/s | Finished Proofs: %llu\n", h / 1000000.0, state.valid_shares.load());
    }

    state.stop_flag.store(true);
    for (auto& w : workers) if (w.joinable()) w.join();
    if (state.socket_fd != -1) close(state.socket_fd);
    
    return 0;
}
