#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 8;
const int HIDDEN_DIM = 32;
const int ALIAS_BINS = 32;
const float GRID_MIN = -5.0f;
const float GRID_MAX = 5.0f;

enum ActType { ALIAS, VANILLA, STOCH_GAUSS, LEAKY, UNIF_NEG1_1, UNIF_ZERO_1 };

struct Params {
    std::vector<float> values; // For Alias
    std::vector<float> m, v;
    std::vector<float> grad;
    Params(int size) : values(size, 0.0f), m(size, 0.0f), v(size, 0.0f), grad(size, 0.0f) {}
};

struct Activation {
    ActType type;
    std::vector<Params> neuron_params;
    std::mt19937 gen;

    Activation(ActType t) : type(t), gen(std::random_device{}()) {
        if (type == ALIAS) {
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                Params p(ALIAS_BINS);
                std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
                for (auto& v : p.values) v = dist(gen);
                neuron_params.push_back(p);
            }
        }
    }

    float evaluate(float x, int neuron_idx) {
        switch (type) {
            case ALIAS: {
                float pos = (x - GRID_MIN) / (GRID_MAX - GRID_MIN) * (ALIAS_BINS - 1);
                int idx = std::clamp((int)pos, 0, ALIAS_BINS - 2);
                float frac = pos - idx;
                return neuron_params[neuron_idx].values[idx] * (1.0f - frac) + neuron_params[neuron_idx].values[idx+1] * frac;
            }
            case VANILLA: return std::max(0.0f, x);
            case STOCH_GAUSS: {
                std::normal_distribution<float> d(std::max(0.0f, x), 0.05f);
                return d(gen);
            }
            case LEAKY: return x > 0 ? x : 0.01f * x;
            case UNIF_NEG1_1: {
                std::uniform_real_distribution<float> d(-1.0f, 1.0f);
                return std::max(0.0f, x) + d(gen) * 0.05f;
            }
            case UNIF_ZERO_1: {
                std::uniform_real_distribution<float> d(0.0f, 1.0f);
                return std::max(0.0f, x) + d(gen) * 0.05f;
            }
        }
        return 0;
    }

    void update_grad(float x, float d_out, int neuron_idx) {
        if (type == ALIAS) {
            float pos = (x - GRID_MIN) / (GRID_MAX - GRID_MIN) * (ALIAS_BINS - 1);
            int idx = std::clamp((int)pos, 0, ALIAS_BINS - 2);
            float frac = pos - idx;
            neuron_params[neuron_idx].grad[idx] += d_out * (1.0f - frac);
            neuron_params[neuron_idx].grad[idx+1] += d_out * frac;
        }
    }
};

struct Problem {
    std::vector<float> coeffs;
    std::vector<int> types; // 0: sin, 1: cos, 2: sq, 3: exp
    
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
            for(int j=0; j<INPUT_DIM; ++j) arg += x[j] * coeffs[i]; // Simple projection
            if(types[i] == 0) val += std::sin(arg);
            else if(types[i] == 1) val += std::cos(arg);
            else if(types[i] == 2) val += arg * arg * 0.1f;
            else val += std::exp(-arg * arg);
        }
        return val;
    }
};

void radam_update(std::vector<float>& p, std::vector<float>& m, std::vector<float>& v, std::vector<float>& g, float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for(size_t i=0; i<p.size(); ++i) {
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

    std::cout << "Running benchmark for: " << name << "..." << std::endl;
    for(size_t p_idx = 0; p_idx < problems.size(); ++p_idx) {
        auto& prob = problems[p_idx];
        Activation act(type);
        std::vector<float> W(INPUT_DIM * HIDDEN_DIM);
        std::vector<float> W_out(HIDDEN_DIM);
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        float scale_h = 1.0f / std::sqrt((float)HIDDEN_DIM);
        float scale_o = 1.0f / (float)HIDDEN_DIM;
        for(auto& w : W) w = dist(gen) * scale_h;
        for(auto& w : W_out) w = dist(gen) * scale_o;

        std::vector<float> m_out(HIDDEN_DIM, 0), v_out(HIDDEN_DIM, 0), g_out(HIDDEN_DIM, 0);

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
                act.update_grad(h_lin[i], d_out * W_out[i], i);
            }

            if(t % 32 == 0) {
                radam_update(W_out, m_out, v_out, g_out, 1e-3f, t/32);
                if(type == ALIAS) {
                    for(int i=0; i<HIDDEN_DIM; ++i) {
                        radam_update(act.neuron_params[i].values, act.neuron_params[i].m, act.neuron_params[i].v, act.neuron_params[i].grad, 1e-3f, t/32);
                    }
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

    results.push_back(run_benchmark(ALIAS, "Alias (32 bins)", problems));
    results.push_back(run_benchmark(VANILLA, "Vanilla (ReLU)", problems));
    results.push_back(run_benchmark(STOCH_GAUSS, "Stoch. Gaussian", problems));
    results.push_back(run_benchmark(LEAKY, "Leaky ReLU", problems));
    results.push_back(run_benchmark(UNIF_NEG1_1, "Uniform [-1, 1]", problems));
    results.push_back(run_benchmark(UNIF_ZERO_1, "Uniform [0, 1]", problems));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Activation Type | Err Reduct/s | Iterations/s\n";
    std::cout << "----------------------------------------------\n";
    for(auto& r : results) {
        std::cout << std::left << std::setw(16) << r.name << " | " 
                  << std::setw(12) << r.err_reduction << " | " 
                  << r.iterations_per_sec << "\n";
    }

    // Output TeX
    std::ofstream tex("benchmark_table.tex");
    tex << "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lrr}\n\\toprule\n";
    tex << "Activation Function & Err Reduct/s & Iterations/s \\\\\n\\midrule\n";
    for(auto& r : results) {
        tex << r.name << " & " << r.err_reduction << " & " << (int)r.iterations_per_sec << " \\\\\n";
    }
    tex << "\\bottomrule\n\\end{tabular}\n\\caption{Activation Function Benchmark (16 problems, 4s each)}\n\\end{table}\n";

    return 0;
}
