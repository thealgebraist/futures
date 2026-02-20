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
const int OUTPUT_DIM = 1;
const int BATCH_SIZE = 32;

struct Parameter {
    std::vector<float> v;    // direction
    std::vector<float> g;    // magnitude
    std::vector<float> grad_v;
    std::vector<float> grad_g;
    std::vector<float> m_v, v_v;
    std::vector<float> m_g, v_g;

    Parameter(int in, int out, float init_scale) {
        int size = in * out;
        v.resize(size);
        g.resize(out);
        grad_v.assign(size, 0.0f);
        grad_g.assign(out, 0.0f);
        m_v.assign(size, 0.0f);
        v_v.assign(size, 0.0f);
        m_g.assign(out, 0.0f);
        v_g.assign(out, 0.0f);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, init_scale);
        for (auto& val : v) val = dist(gen);
        
        for (int j = 0; j < out; ++j) {
            float norm = 0;
            vDSP_dotpr(v.data() + j * in, 1, v.data() + j * in, 1, &norm, in);
            g[j] = std::sqrt(norm) + 1e-8f;
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

struct ScalingModel {
    int width;
    Parameter W1, W_out;
    std::vector<float> w1_flat, wout_flat;

    ScalingModel(int n) : 
        width(n),
        W1(INPUT_DIM, n, 1.0f / std::sqrt((float)n)), // muP hidden
        W_out(n, OUTPUT_DIM, 1.0f / (float)n),        // muP output
        w1_flat(INPUT_DIM * n),
        wout_flat(n * OUTPUT_DIM) {}

    float forward(const std::vector<float>& x) {
        W1.compute_weights(w1_flat, INPUT_DIM, width);
        W_out.compute_weights(wout_flat, width, OUTPUT_DIM);

        std::vector<float> h(width);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, width, INPUT_DIM, 1.0f, w1_flat.data(), INPUT_DIM, x.data(), 1, 0.0f, h.data(), 1);
        for(auto& v : h) v = std::tanh(v);

        float y = 0;
        vDSP_dotpr(h.data(), 1, wout_flat.data(), 1, &y, width);
        return y;
    }
};

std::vector<float> load_data(const std::string& path) {
    std::vector<float> data;
    std::ifstream file(path);
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while(std::getline(ss, cell, ',')) row.push_back(cell);
        if(!row.empty()) {
            try { data.push_back(std::stof(row[0])); } catch(...) {}
        }
    }
    return data;
}

int main() {
    auto prices = load_data("data/futures_16_changes.csv");
    if (prices.size() < 100) return 1;

    int train_split = prices.size() * 0.8;
    
    for (int n = 1; n <= 16; ++n) {
        ScalingModel model(n);
        auto start_time = std::chrono::steady_clock::now();
        int t = 1;

        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 20) {
            int idx = rand() % (train_split - INPUT_DIM - 1);
            std::vector<float> x(INPUT_DIM);
            for(int j=0; j<INPUT_DIM; ++j) x[j] = prices[idx+j];
            float target = prices[idx+INPUT_DIM];

            float pred = model.forward(x);
            float diff = pred - target;

            // Grads
            for(int i=0; i<INPUT_DIM * n; ++i) model.W1.grad_v[i] += diff * 0.001f;
            for(int i=0; i<n; ++i) model.W1.grad_g[i] += diff * 0.001f;
            for(int i=0; i<n; ++i) model.W_out.grad_v[i] += diff * 0.001f;
            model.W_out.grad_g[0] += diff * 0.001f;

            radam_step(model.W1.v, model.W1.grad_v, model.W1.m_v, model.W1.v_v, 1e-3f, t);
            radam_step(model.W1.g, model.W1.grad_g, model.W1.m_g, model.W1.v_g, 1e-3f, t);
            radam_step(model.W_out.v, model.W_out.grad_v, model.W_out.m_v, model.W_out.v_v, 1e-3f, t);
            radam_step(model.W_out.g, model.W_out.grad_g, model.W_out.m_g, model.W_out.v_g, 1e-3f, t);
            t++;
        }

        // Eval
        float mse = 0;
        int count = 0;
        for (int i = train_split; i < (int)prices.size() - INPUT_DIM - 1; ++i) {
            std::vector<float> x(INPUT_DIM);
            for(int j=0; j<INPUT_DIM; ++j) x[j] = prices[i+j];
            float p = model.forward(x);
            float d = p - prices[i+INPUT_DIM];
            mse += d*d;
            count++;
        }
        std::cout << n << " " << (mse / count) << std::endl;
    }

    return 0;
}
