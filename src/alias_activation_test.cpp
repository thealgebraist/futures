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
const int HIDDEN_DIM = 32;
const int ALIAS_SIZE = 512;
const float GRID_MIN = -5.0f;
const float GRID_MAX = 5.0f;

struct AliasAct {
    std::vector<float> values;
    std::vector<float> grad;
    std::vector<float> m, v;

    AliasAct() : values(ALIAS_SIZE), grad(ALIAS_SIZE, 0.0f), m(ALIAS_SIZE, 0.0f), v(ALIAS_SIZE, 0.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
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

int main(int argc, char** argv) {
    if(argc < 2) return 1;
    int mode = std::stoi(argv[1]);

    std::vector<float> W(INPUT_DIM * HIDDEN_DIM);
    std::vector<float> g_wn(HIDDEN_DIM, 1.0f); // Weight Norm gains
    std::random_device rd; std::mt19937 gen(rd());
    
    // muP scaling: hidden layers by 1/sqrt(width)
    float scale = 1.0f / std::sqrt((float)HIDDEN_DIM);

    if(mode == 1) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for(auto& w : W) w = dist(gen) * scale;
    } else if(mode == 2) {
        for(int i=0; i<HIDDEN_DIM; ++i) {
            for(int j=0; j<INPUT_DIM; ++j) {
                W[i*INPUT_DIM+j] = std::cos(2.0 * M_PI * i * j / INPUT_DIM) * scale;
            }
        }
    } else if(mode == 3) {
        for(int i=0; i<HIDDEN_DIM; ++i) {
            for(int j=0; j<INPUT_DIM; ++j) {
                W[i*INPUT_DIM+j] = ((j < (INPUT_DIM/(i+1))) ? 1.0f : -1.0f) * scale;
            }
        }
    }

    std::vector<AliasAct> acts(HIDDEN_DIM);
    // muP scaling: output layer by 1/width
    std::vector<float> W_out(HIDDEN_DIM);
    for(auto& w : W_out) w = 0.1f / HIDDEN_DIM;

    auto start = std::chrono::steady_clock::now();
    int t = 1;
    while(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < 120) {
        std::normal_distribution<float> d_dist(0.0f, 1.0f);
        float x[INPUT_DIM];
        float norm_sq = 0;
        for(int i=0; i<INPUT_DIM; ++i) {
            x[i] = d_dist(gen);
            norm_sq += x[i]*x[i];
        }
        // Target: 20D Sphere approximation
        float target = (norm_sq < 1.0f) ? 1.0f : 0.0f;

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

        float d_out = pred - target;
        for(int i=0; i<HIDDEN_DIM; ++i) {
            acts[i].update_grad(h_lin[i], d_out * W_out[i]);
            radam_update(acts[i].values, acts[i].m, acts[i].v, acts[i].grad, 1e-3f, t);
        }
        t++;
    }

    std::string name = (mode==1?"rand":(mode==2?"fft":"haar"));
    std::ofstream out("curves_" + name + ".txt");
    for(int i=0; i<ALIAS_SIZE; ++i) {
        float x = GRID_MIN + (float)i/(ALIAS_SIZE-1) * (GRID_MAX - GRID_MIN);
        out << x << " ";
        for(int j=0; j<HIDDEN_DIM; ++j) out << acts[j].values[i] << (j==HIDDEN_DIM-1?"":" ");
        out << "\n";
    }

    return 0;
}
