#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <arm_neon.h>
#include "levy_kernels.hpp"

using namespace levy;

struct Matrix {
    int rows, cols;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<float> m, v;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c), grad(r * c, 0.0f), m(r * c, 0.0f), v(r * c, 0.0f) {}

    static Matrix random(int r, int c) {
        Matrix mat(r, c);
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = std::sqrt(6.0f / (r + c));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& val : mat.data) val = dist(gen);
        return mat;
    }
};

void neon_gemv(int r, int c, const float* A, const float* x, float* y) {
    for (int i = 0; i < r; ++i) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (int j = 0; j < c; j += 4) {
            float32x4_t a_vec = vld1q_f32(A + i * c + j);
            float32x4_t x_vec = vld1q_f32(x + j);
            sum_vec = vfmaq_f32(sum_vec, a_vec, x_vec);
        }
        y[i] = vaddvq_f32(sum_vec);
    }
}

void radam_step(Matrix& p, float lr, int t) {
    float norm_sq = 0;
    for (float g : p.grad) norm_sq += g * g;
    float norm = std::sqrt(norm_sq);
    if (norm > CLIP_VAL) {
        float scale = CLIP_VAL / (norm + 1e-9f);
        for (float& g : p.grad) g *= scale;
    }
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for (size_t i = 0; i < p.data.size(); ++i) {
        p.m[i] = beta1 * p.m[i] + (1.0f - beta1) * p.grad[i];
        p.v[i] = beta2 * p.v[i] + (1.0f - beta2) * p.grad[i] * p.grad[i];
        float m_hat = p.m[i] / (1.0f - std::pow(beta1, (float)t));
        float v_hat = std::sqrt(p.v[i] / (1.0f - std::pow(beta2, (float)t)));
        float update = lr * m_hat / (v_hat + eps);
        if(!std::isnan(update)) p.data[i] -= update;
        p.grad[i] = 0;
    }
}

struct Model {
    Matrix W1, b1, W2, b2, W_out, b_out;
    std::mt19937 gen;

    Model() : 
        W1(Matrix::random(HIDDEN_DIM, INPUT_DIM)), b1(HIDDEN_DIM, 1),
        W2(Matrix::random(HIDDEN_DIM, HIDDEN_DIM)), b2(HIDDEN_DIM, 1),
        W_out(Matrix::random(OUTPUT_DIM, HIDDEN_DIM)), b_out(OUTPUT_DIM, 1) {
        std::random_device rd;
        gen.seed(rd());
    }

    void forward_single(const float* x, float* y, bool training) {
        float h1[HIDDEN_DIM], h2[HIDDEN_DIM];
        neon_gemv(HIDDEN_DIM, INPUT_DIM, W1.data.data(), x, h1);
        for(int i=0; i<HIDDEN_DIM; ++i) {
            h1[i] = std::tanh(h1[i] + b1.data[i]);
            if(training) h1[i] += generate_stable(gen);
        }
        neon_gemv(HIDDEN_DIM, HIDDEN_DIM, W2.data.data(), h1, h2);
        for(int i=0; i<HIDDEN_DIM; ++i) {
            h2[i] = std::tanh(h2[i] + b2.data[i]);
            if(training) h2[i] += generate_stable(gen);
        }
        neon_gemv(OUTPUT_DIM, HIDDEN_DIM, W_out.data.data(), h2, y);
        y[0] += b_out.data[0];
    }
};

std::vector<float> load_csv(const std::string& path) {
    std::vector<float> data;
    std::ifstream file(path);
    std::string line;
    if(!std::getline(file, line)) return data;
    int close_idx = -1;
    std::stringstream ss_head(line);
    std::string head;
    int cur = 0;
    while(std::getline(ss_head, head, ',')) {
        if(head == "Close") close_idx = cur;
        cur++;
    }
    if(close_idx == -1) close_idx = 1;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while(std::getline(ss, cell, ',')) row.push_back(cell);
        if((int)row.size() > close_idx) {
            try { data.push_back(std::stof(row[close_idx])); } catch(...) {}
        }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    std::string path = argv[1];
    std::string symbol = argv[2];
    int duration = std::stoi(argv[3]);

    auto prices = load_csv(path);
    if (prices.size() < 100) return 1;
    std::vector<float> rets;
    for(size_t i=1; i<prices.size(); ++i) rets.push_back((prices[i] - prices[i-1]) / (prices[i-1] + 1e-9f));
    float mean=0, std=0;
    for(auto r : rets) mean += r; mean /= rets.size();
    for(auto r : rets) std += (r-mean)*(r-mean); std = std::sqrt(std/rets.size());
    for(auto& r : rets) r = (r-mean)/(std + 1e-6f);

    int train_split = rets.size() * 0.8;
    Model model;
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < duration) {
        for(int b=0; b<BATCH_SIZE; ++b) {
            int idx = rand() % (train_split - INPUT_DIM - 1);
            float bx[INPUT_DIM], by, pred;
            for(int j=0; j<INPUT_DIM; ++j) bx[j] = rets[idx+j];
            by = rets[idx+INPUT_DIM];
            model.forward_single(bx, &pred, true);
            float d = pred - by;
            if(std::abs(d) > 10.0f) d = (d > 0) ? 10.0f : -10.0f;
            for(int i=0; i<model.W1.grad.size(); ++i) model.W1.grad[i] += d * 0.0001f;
            for(int i=0; i<model.W_out.grad.size(); ++i) model.W_out.grad[i] += d * 0.0001f;
        }
        radam_step(model.W1, 1e-4f, t);
        radam_step(model.b1, 1e-4f, t);
        radam_step(model.W2, 1e-4f, t);
        radam_step(model.b2, 1e-4f, t);
        radam_step(model.W_out, 1e-4f, t);
        radam_step(model.b_out, 1e-4f, t);
        t++;
    }
    int test_n = rets.size() - train_split - INPUT_DIM - 1;
    float total_mse = 0; int count = 0;
    for(int i=0; i<test_n; ++i) {
        float bx[INPUT_DIM], pred;
        for(int j=0; j<INPUT_DIM; ++j) bx[j] = rets[train_split+i+j];
        model.forward_single(bx, &pred, false);
        float d = pred - rets[train_split+i+INPUT_DIM];
        if(!std::isnan(d)) { total_mse += d*d; count++; }
    }
    std::cout << symbol << " NEON " << (count > 0 ? total_mse/count : 1.0f) << " " << t << std::endl;
    return 0;
}
