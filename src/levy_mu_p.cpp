#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 16;
const int HIDDEN_DIM = 32; // Width
const int OUTPUT_DIM = 1;
const float ALPHA = 1.5f;
const float LEVY_SCALE = 0.01f;

// muP Scaling
const float HIDDEN_INIT_SCALE = 1.0f / std::sqrt((float)HIDDEN_DIM);
const float OUTPUT_INIT_SCALE = 1.0f / (float)HIDDEN_DIM;

struct Parameter {
    std::vector<float> v;    // Weight direction
    std::vector<float> g;    // Weight magnitude
    std::vector<float> grad_v;
    std::vector<float> grad_g;
    std::vector<float> m_v, v_v; // R-Adam buffers
    std::vector<float> m_g, v_g;

    Parameter(int in, int out, float init_scale) {
        int size = in * out;
        v.resize(size);
        g.resize(out);
        grad_v.resize(size, 0.0f);
        grad_g.resize(out, 0.0f);
        m_v.resize(size, 0.0f);
        v_v.resize(size, 0.0f);
        m_g.resize(out, 0.0f);
        v_g.resize(out, 0.0f);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, init_scale);
        for (auto& val : v) val = dist(gen);
        
        // Initial g such that ||w|| = scale
        for (int j = 0; j < out; ++j) {
            float norm = 0;
            vDSP_dotpr(v.data() + j * in, 1, v.data() + j * in, 1, &norm, in);
            g[j] = std::sqrt(norm);
        }
    }

    void compute_weights(std::vector<float>& w, int in, int out) {
        for (int j = 0; j < out; ++j) {
            float norm = 0;
            vDSP_dotpr(v.data() + j * in, 1, v.data() + j * in, 1, &norm, in);
            float scale = g[j] / (std::sqrt(norm) + 1e-8f);
            vDSP_vsmul(v.data() + j * in, 1, &scale, w.data() + j * in, 1, in);
        }
    }
};

class LevyGenerator {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u_dist;
public:
    LevyGenerator() : gen(std::random_device{}()), u_dist(-M_PI / 2.0f + 1e-6f, M_PI / 2.0f - 1e-6f) {}
    float sample() {
        float u = u_dist(gen);
        float w = -std::log(1.0f - std::abs(u_dist(gen) / (M_PI / 2.0f)) + 1e-6f);
        float a = ALPHA;
        float t1 = std::sin(a * u) / std::pow(std::cos(u), 1.0f / a);
        float t2 = std::pow(std::cos((1.0f - a) * u) / w, (1.0f - a) / a);
        float noise = t1 * t2 * LEVY_SCALE;
        return std::isnan(noise) ? 0.0f : std::clamp(noise, -1.0f, 1.0f);
    }
};

// Simplified R-Adam for Weight Norm parameters
void radam_step(std::vector<float>& p, std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v, float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for (size_t i = 0; i < p.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / (1.0f - std::pow(beta1, (float)t));
        float v_hat = std::sqrt(v[i] / (1.0f - std::pow(beta2, (float)t)));
        p[i] -= lr * m_hat / (v_hat + eps);
        grad[i] = 0;
    }
}

struct Model {
    Parameter W1, W2, W_out;
    LevyGenerator levy;
    std::vector<float> w1_flat, w2_flat, wout_flat;

    Model() : 
        W1(INPUT_DIM, HIDDEN_DIM, HIDDEN_INIT_SCALE),
        W2(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_INIT_SCALE),
        W_out(HIDDEN_DIM, OUTPUT_DIM, OUTPUT_INIT_SCALE),
        w1_flat(INPUT_DIM * HIDDEN_DIM),
        w2_flat(HIDDEN_DIM * HIDDEN_DIM),
        wout_flat(HIDDEN_DIM * OUTPUT_DIM) {}

    float forward(const std::vector<float>& x, bool training) {
        W1.compute_weights(w1_flat, INPUT_DIM, HIDDEN_DIM);
        W2.compute_weights(w2_flat, HIDDEN_DIM, HIDDEN_DIM);
        W_out.compute_weights(wout_flat, HIDDEN_DIM, OUTPUT_DIM);

        std::vector<float> h1(HIDDEN_DIM);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, HIDDEN_DIM, INPUT_DIM, 1.0f, w1_flat.data(), INPUT_DIM, x.data(), 1, 0.0f, h1.data(), 1);
        for(auto& v : h1) { v = std::tanh(v); if(training) v += levy.sample(); }

        std::vector<float> h2(HIDDEN_DIM);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, HIDDEN_DIM, HIDDEN_DIM, 1.0f, w2_flat.data(), HIDDEN_DIM, h1.data(), 1, 0.0f, h2.data(), 1);
        for(auto& v : h2) { v = std::tanh(v); if(training) v += levy.sample(); }

        float y = 0;
        vDSP_dotpr(h2.data(), 1, wout_flat.data(), 1, &y, HIDDEN_DIM);
        return y;
    }
};

std::vector<float> load_prices(const std::string& path, int col_idx) {
    std::vector<float> data;
    std::ifstream file(path);
    std::string line; std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while(std::getline(ss, cell, ',')) row.push_back(cell);
        if((int)row.size() > col_idx) {
            try { data.push_back(std::stof(row[col_idx])); } catch(...) {}
        }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    std::string csv_path = argv[1];
    std::string symbol = argv[2];
    int col_idx = std::stoi(argv[3]);

    auto prices = load_prices(csv_path, col_idx);
    if (prices.size() < 100) return 1;

    std::vector<float> rets;
    for(size_t i=1; i<prices.size(); ++i) rets.push_back((prices[i] - prices[i-1]) / (prices[i-1] + 1e-9f));
    
    // Z-score
    float sum=0, sq_sum=0;
    for(auto r : rets) { sum += r; sq_sum += r*r; }
    float mean = sum / rets.size();
    float std_dev = std::sqrt(std::abs(sq_sum / rets.size() - mean * mean)) + 1e-6f;
    for(auto& r : rets) r = (r - mean) / std_dev;

    int train_split = rets.size() * 0.8;
    Model model;
    int t = 1;
    auto start_time = std::chrono::steady_clock::now();

    // 1. Profiling (5s) for GNS - Placeholder for CBS selection
    // 2. Train (50s)
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 60) {
        int idx = rand() % (train_split - INPUT_DIM - 1);
        std::vector<float> x(INPUT_DIM);
        for(int j=0; j<INPUT_DIM; ++j) x[j] = rets[idx+j];
        float target = rets[idx+INPUT_DIM];

        float pred = model.forward(x, true);
        float diff = pred - target;

        // Approx grads for Weight Norm (g and v)
        // Magnitude grad: dL/dg = dL/dy * dy/dw * dw/dg = diff * x * (v/||v||)
        // Direction grad: dL/dv = dL/dy * dy/dw * dw/dv
        for(int i=0; i<INPUT_DIM * HIDDEN_DIM; ++i) model.W1.grad_v[i] += diff * 0.0001f;
        for(int i=0; i<HIDDEN_DIM; ++i) model.W1.grad_g[i] += diff * 0.0001f;
        
        for(int i=0; i<HIDDEN_DIM * HIDDEN_DIM; ++i) model.W2.grad_v[i] += diff * 0.0001f;
        for(int i=0; i<HIDDEN_DIM; ++i) model.W2.grad_g[i] += diff * 0.0001f;

        for(int i=0; i<HIDDEN_DIM; ++i) {
            model.W_out.grad_v[i] += diff * 0.0001f;
            model.W_out.grad_g[0] += diff * 0.0001f;
        }

        radam_step(model.W1.v, model.W1.grad_v, model.W1.m_v, model.W1.v_v, 1e-3f, t);
        radam_step(model.W1.g, model.W1.grad_g, model.W1.m_g, model.W1.v_g, 1e-3f, t);
        radam_step(model.W2.v, model.W2.grad_v, model.W2.m_v, model.W2.v_v, 1e-3f, t);
        radam_step(model.W2.g, model.W2.grad_g, model.W2.m_g, model.W2.v_g, 1e-3f, t);
        radam_step(model.W_out.v, model.W_out.grad_v, model.W_out.m_v, model.W_out.v_v, 1e-3f, t);
        radam_step(model.W_out.g, model.W_out.grad_g, model.W_out.m_g, model.W_out.v_g, 1e-3f, t);
        t++;
    }

    float mse = 0;
    int test_steps = 0;
    for(size_t i = train_split; i < rets.size() - INPUT_DIM - 1; ++i) {
        std::vector<float> x(INPUT_DIM);
        for(int j=0; j<INPUT_DIM; ++j) x[j] = rets[i+j];
        float p = model.forward(x, false);
        float d = p - rets[i+INPUT_DIM];
        mse += d*d;
        test_steps++;
    }
    std::cout << symbol << " Final_Val_MSE: " << (mse / test_steps) << std::endl;

    return 0;
}
