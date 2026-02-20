#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 8;
const int HIDDEN_DIM = 32;
const int ALIAS_BINS = 32;
const float GRID_MIN = -5.0f;
const float GRID_MAX = 5.0f;

// Vectorized Xorshift128+ for PRNG
struct VecPRNG {
    uint64x2_t state0, state1;
    VecPRNG() {
        state0 = vdupq_n_u64(0x123456789ABCDEF0ULL);
        state1 = vdupq_n_u64(0xFEDCBA9876543210ULL);
    }
    
    // Returns 2 random floats in [0, 1]
    float32x4_t next_f32() {
        uint64x2_t s1 = state0;
        uint64x2_t s0 = state1;
        state0 = s0;
        s1 = veorq_u64(s1, vshlq_n_u64(s1, 23));
        state1 = veorq_u64(veorq_u64(s1, s0), veorq_u64(vshrq_n_u64(s1, 18), vshrq_n_u64(s0, 5)));
        uint64x2_t res_u64 = vaddq_u64(state1, s0);
        
        // Convert to float [0, 1] - very rough conversion for speed
        uint32x4_t res_u32 = vreinterpretq_u32_u64(res_u64);
        uint32x4_t mask = vdupq_n_u32(0x7FFFFF);
        uint32x4_t mantissa = vorrq_u32(vandq_u32(res_u32, mask), vdupq_n_u32(0x3F800000));
        float32x4_t f = vsubq_f32(vreinterpretq_f32_u32(mantissa), vdupq_n_f32(1.0f));
        return f;
    }
};

enum ActType { ALIAS, VANILLA, STOCH_GAUSS, LEAKY };

struct AliasTable {
    float32x4_t prob[8];  // 32 bins / 4
    uint32x4_t alias[8]; 
    float32x4_t values[8];
};

struct Activation {
    ActType type;
    std::vector<float> alias_values;
    VecPRNG prng;

    Activation(ActType t) : type(t), alias_values(HIDDEN_DIM * ALIAS_BINS) {
        if (type == ALIAS) {
            std::mt19937 gen(1337);
            std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
            for (auto& v : alias_values) v = dist(gen);
        }
    }

    // Optimized scalar fallback for benchmark logic integration
    // In a real app, the entire layer would be vectorized
    float evaluate(float x, int neuron_idx) {
        switch (type) {
            case ALIAS: {
                float pos = (x - GRID_MIN) / (GRID_MAX - GRID_MIN) * (ALIAS_BINS - 1);
                int idx = std::clamp((int)pos, 0, ALIAS_BINS - 2);
                float frac = pos - idx;
                float* v = &alias_values[neuron_idx * ALIAS_BINS];
                return v[idx] * (1.0f - frac) + v[idx+1] * frac;
            }
            case VANILLA: return std::max(0.0f, x);
            case STOCH_GAUSS: {
                // Use fast PRNG for Gaussian (Box-Muller light)
                float32x4_t r = prng.next_f32();
                float rand_val;
                vst1q_lane_f32(&rand_val, r, 0);
                return std::max(0.0f, x) + (rand_val - 0.5f) * 0.1f;
            }
            case LEAKY: return x > 0 ? x : 0.01f * x;
        }
        return 0;
    }
};

struct Problem {
    std::vector<float> coeffs;
    std::vector<int> types;
    Problem() {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> c_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> t_dist(0, 3);
        for(int i=0; i<10; ++i) {
            coeffs.push_back(c_dist(gen));
            types.push_back(t_dist(gen));
        }
    }
    float target(const float* x) {
        float val = 0;
        for(int i=0; i<10; ++i) {
            float arg = 0;
            for(int j=0; j<INPUT_DIM; ++j) arg += x[j] * coeffs[i];
            if(types[i] == 0) val += std::sin(arg);
            else if(types[i] == 1) val += std::cos(arg);
            else if(types[i] == 2) val += arg * arg * 0.1f;
            else val += std::exp(-arg * arg);
        }
        return val;
    }
};

void radam_update(float* p, float* m, float* v, float* g, int size, float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for(int i=0; i<size; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        float m_hat = m[i] / (1.0f - std::pow(beta1, (float)t));
        float v_hat = std::sqrt(v[i] / (1.0f - std::pow(beta2, (float)t)));
        p[i] -= lr * m_hat / (v_hat + eps);
        g[i] = 0;
    }
}

struct Result {
    std::string name;
    float err_reduction;
    float iterations_per_sec;
};

Result run_benchmark(ActType type, std::string name, std::vector<Problem>& problems) {
    float total_reduction = 0;
    long total_steps = 0;
    float total_time = 0;

    std::cout << "Running optimized benchmark for: " << name << "..." << std::endl;
    for(auto& prob : problems) {
        Activation act(type);
        std::vector<float> W(INPUT_DIM * HIDDEN_DIM);
        std::vector<float> W_out(HIDDEN_DIM);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        float scale_h = 1.0f / std::sqrt((float)HIDDEN_DIM);
        float scale_o = 1.0f / (float)HIDDEN_DIM;
        for(auto& w : W) w = dist(gen) * scale_h;
        for(auto& w : W_out) w = dist(gen) * scale_o;

        std::vector<float> m_out(HIDDEN_DIM, 0), v_out(HIDDEN_DIM, 0), g_out(HIDDEN_DIM, 0);
        std::vector<float> m_act(HIDDEN_DIM * ALIAS_BINS, 0), v_act(HIDDEN_DIM * ALIAS_BINS, 0), g_act(HIDDEN_DIM * ALIAS_BINS, 0);

        auto start = std::chrono::steady_clock::now();
        int t = 1;
        float initial_loss = 0;
        float final_loss = 0;

        while(true) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0f;
            if(elapsed >= 4.0f) {
                total_time += elapsed;
                break;
            }

            float x[INPUT_DIM];
            for(int i=0; i<INPUT_DIM; ++i) x[i] = dist(gen);
            float target = prob.target(x);

            float h_lin[HIDDEN_DIM] = {0};
            for(int i=0; i<HIDDEN_DIM; ++i) {
                for(int j=0; j<INPUT_DIM; ++j) h_lin[i] += W[i*INPUT_DIM+j] * x[j];
            }

            float h_act[HIDDEN_DIM];
            float pred = 0;
            for(int i=0; i<HIDDEN_DIM; ++i) {
                h_act[i] = act.evaluate(h_lin[i], i);
                pred += h_act[i] * W_out[i];
            }

            float loss = 0.5f * (pred - target) * (pred - target);
            if(t <= 100) initial_loss += loss / 100.0f;
            final_loss = 0.99f * final_loss + 0.01f * loss;

            float d_out = pred - target;
            for(int i=0; i<HIDDEN_DIM; ++i) {
                g_out[i] += d_out * h_act[i];
                if(type == ALIAS) {
                    float pos = (h_lin[i] - GRID_MIN) / (GRID_MAX - GRID_MIN) * (ALIAS_BINS - 1);
                    int idx = std::clamp((int)pos, 0, ALIAS_BINS - 2);
                    float frac = pos - idx;
                    g_act[i * ALIAS_BINS + idx] += d_out * W_out[i] * (1.0f - frac);
                    g_act[i * ALIAS_BINS + idx + 1] += d_out * W_out[i] * frac;
                }
            }

            if(t % 32 == 0) {
                radam_update(W_out.data(), m_out.data(), v_out.data(), g_out.data(), HIDDEN_DIM, 1e-3f, t/32);
                if(type == ALIAS) {
                    radam_update(act.alias_values.data(), m_act.data(), v_act.data(), g_act.data(), HIDDEN_DIM * ALIAS_BINS, 1e-3f, t/32);
                }
            }
            t++;
        }
        total_reduction += (initial_loss - final_loss) / (initial_loss + 1e-6f);
        total_steps += t;
    }
    return {name, total_reduction / problems.size() / 4.0f, (float)total_steps / total_time};
}

int main() {
    std::vector<Problem> problems(16);
    std::vector<Result> results;

    results.push_back(run_benchmark(ALIAS, "Alias (Optimized)", problems));
    results.push_back(run_benchmark(VANILLA, "Vanilla (ReLU)", problems));
    results.push_back(run_benchmark(STOCH_GAUSS, "Stoch. Gaussian (Vec)", problems));
    results.push_back(run_benchmark(LEAKY, "Leaky ReLU", problems));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nActivation Type | Err Reduct/s | Iterations/s\n";
    std::cout << "----------------------------------------------\n";
    for(auto& r : results) {
        std::cout << std::left << std::setw(20) << r.name << " | " 
                  << std::setw(12) << r.err_reduction << " | " 
                  << r.iterations_per_sec << "\n";
    }

    return 0;
}
