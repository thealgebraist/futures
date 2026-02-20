#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 20;
const int HIDDEN_DIM = 2;
const int ALIAS_SIZE = 64;
const float GRID_MIN = -5.0f;
const float GRID_MAX = 5.0f;

struct AliasAct {
    std::vector<float> values;
    std::vector<float> grad;
    std::vector<float> m, v;

    AliasAct() : values(ALIAS_SIZE), grad(ALIAS_SIZE, 0.0f), m(ALIAS_SIZE, 0.0f), v(ALIAS_SIZE, 0.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for(auto& v : values) v = dist(gen);
    }

    float evaluate(float x) {
        float pos = (x - GRID_MIN) / (GRID_MAX - GRID_MIN) * (ALIAS_SIZE - 1);
        int idx = std::clamp((int)pos, 0, ALIAS_SIZE - 2);
        float frac = pos - idx;
        return values[idx] * (1.0f - frac) + values[idx+1] * frac;
    }

    void update_grad(float x, float d_out) {
        float pos = (x - GRID_MIN) / (GRID_MAX - GRID_MIN) * (ALIAS_SIZE - 1);
        int idx = std::clamp((int)pos, 0, ALIAS_SIZE - 2);
        float frac = pos - idx;
        grad[idx] += d_out * (1.0f - frac);
        grad[idx+1] += d_out * frac;
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

int main() {
    std::vector<float> W(INPUT_DIM * HIDDEN_DIM);
    std::random_device rd; std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    float scale = 1.0f / std::sqrt((float)HIDDEN_DIM);
    for(auto& w : W) w = dist(gen) * scale;

    std::vector<AliasAct> acts(HIDDEN_DIM);
    std::vector<float> W_out(HIDDEN_DIM);
    for(auto& w : W_out) w = 1.0f / HIDDEN_DIM;

    auto start = std::chrono::steady_clock::now();
    int t = 1;
    float running_loss = 0;
    while(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < 120) {
        float x[INPUT_DIM];
        float norm_sq = 0;
        for(int i=0; i<INPUT_DIM; ++i) {
            x[i] = dist(gen) * 0.5f; // Sample closer to origin
            norm_sq += x[i]*x[i];
        }
        float target = (norm_sq < 0.25f) ? 1.0f : 0.0f;

        float h_lin[HIDDEN_DIM] = {0};
        for(int i=0; i<HIDDEN_DIM; ++i) {
            for(int j=0; j<INPUT_DIM; ++j) h_lin[i] += W[i*INPUT_DIM+j] * x[j];
        }

        float h_act[HIDDEN_DIM];
        float pred = 0;
        for(int i=0; i<HIDDEN_DIM; ++i) {
            h_act[i] = acts[i].evaluate(h_lin[i]);
            pred += h_act[i] * W_out[i];
        }

        float loss = 0.5f * (pred - target) * (pred - target);
        running_loss = 0.999f * running_loss + 0.001f * loss;
        if(t % 100000 == 0) std::cout << "Step " << t << " Loss: " << running_loss << std::endl;

        float d_out = pred - target;
        for(int i=0; i<HIDDEN_DIM; ++i) {
            acts[i].update_grad(h_lin[i], d_out * W_out[i]);
            radam_update(acts[i].values, acts[i].m, acts[i].v, acts[i].grad, 1e-2f, t);
        }
        t++;
    }

    // Save curves
    std::ofstream out_c("sphere_curves.txt");
    for(int i=0; i<ALIAS_SIZE; ++i) {
        float x = GRID_MIN + (float)i/(ALIAS_SIZE-1) * (GRID_MAX - GRID_MIN);
        out_c << x << " ";
        for(int j=0; j<HIDDEN_DIM; ++j) out_c << acts[j].values[i] << (j==HIDDEN_DIM-1?"":" ");
        out_c << "\n";
    }

    // Save activation map (2D slice)
    std::ofstream out_m("sphere_map.txt");
    int grid_res = 100;
    for(int i=0; i<grid_res; ++i) {
        float x1 = -2.0f + 4.0f * i / (grid_res - 1);
        for(int j=0; j<grid_res; ++j) {
            float x2 = -2.0f + 4.0f * j / (grid_res - 1);
            
            float input[INPUT_DIM] = {0};
            input[0] = x1; input[1] = x2;
            
            float h_lin[HIDDEN_DIM] = {0};
            for(int k=0; k<HIDDEN_DIM; ++k) {
                for(int l=0; l<INPUT_DIM; ++l) h_lin[k] += W[k*INPUT_DIM+l] * input[l];
            }
            
            float pred = 0;
            for(int k=0; k<HIDDEN_DIM; ++k) {
                pred += acts[k].evaluate(h_lin[k]) * W_out[k];
            }
            out_m << pred << (j==grid_res-1?"":" ");
        }
        out_m << "\n";
    }

    return 0;
}
