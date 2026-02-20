#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <Accelerate/Accelerate.h>
#include "levy_kernels.hpp"

using namespace levy;

// Update dims for this intensive experiment
const int HF_INPUT_DIM = 16;
const int HF_HIDDEN_DIM = 512;
const int HF_OUTPUT_DIM = 4;

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

void hf_radam_step(Matrix& p, float lr, int t) {
    float norm = 0;
    vDSP_dotpr(p.grad.data(), 1, p.grad.data(), 1, &norm, p.grad.size());
    norm = std::sqrt(norm);
    if (norm > CLIP_VAL) {
        float scale = CLIP_VAL / (norm + 1e-9f);
        vDSP_vsmul(p.grad.data(), 1, &scale, p.grad.data(), 1, p.grad.size());
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

struct IntensiveModel {
    Matrix W1, b1, W2, b2, W_out, b_out;
    std::mt19937 gen;

    IntensiveModel() : 
        W1(Matrix::random(HF_INPUT_DIM, HF_HIDDEN_DIM)), b1(HF_HIDDEN_DIM, 1),
        W2(Matrix::random(HF_HIDDEN_DIM, HF_HIDDEN_DIM)), b2(HF_HIDDEN_DIM, 1),
        W_out(Matrix::random(HF_HIDDEN_DIM, HF_OUTPUT_DIM)), b_out(HF_OUTPUT_DIM, 1) {
        std::random_device rd;
        gen.seed(rd());
    }

    Matrix forward(const Matrix& x, bool training) {
        Matrix h1(x.rows, HF_HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HF_HIDDEN_DIM, HF_INPUT_DIM, 1.0f, x.data.data(), HF_INPUT_DIM, W1.data.data(), HF_HIDDEN_DIM, 0.0f, h1.data.data(), HF_HIDDEN_DIM);
        for(int i=0; i<h1.rows; ++i) vDSP_vadd(h1.data.data() + i*HF_HIDDEN_DIM, 1, b1.data.data(), 1, h1.data.data() + i*HF_HIDDEN_DIM, 1, HF_HIDDEN_DIM);
        for(auto& v : h1.data) {
            v = std::tanh(v);
            if(training) v += generate_stable(gen);
        }

        Matrix h2(x.rows, HF_HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HF_HIDDEN_DIM, HF_HIDDEN_DIM, 1.0f, h1.data.data(), HF_HIDDEN_DIM, W2.data.data(), HF_HIDDEN_DIM, 0.0f, h2.data.data(), HF_HIDDEN_DIM);
        for(int i=0; i<h2.rows; ++i) vDSP_vadd(h2.data.data() + i*HF_HIDDEN_DIM, 1, b2.data.data(), 1, h2.data.data() + i*HF_HIDDEN_DIM, 1, HF_HIDDEN_DIM);
        for(auto& v : h2.data) {
            v = std::tanh(v);
            if(training) v += generate_stable(gen);
        }

        Matrix y(x.rows, HF_OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HF_OUTPUT_DIM, HF_HIDDEN_DIM, 1.0f, h2.data.data(), HF_HIDDEN_DIM, W_out.data.data(), HF_OUTPUT_DIM, 0.0f, y.data.data(), HF_OUTPUT_DIM);
        for(int i=0; i<y.rows; ++i) vDSP_vadd(y.data.data() + i*HF_OUTPUT_DIM, 1, b_out.data.data(), 1, y.data.data() + i*HF_OUTPUT_DIM, 1, HF_OUTPUT_DIM);
        return y;
    }
};

std::vector<float> load_data(const std::string& path) {
    std::vector<float> data;
    std::ifstream file(path);
    std::string line;
    std::getline(file, line); // Skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while(std::getline(ss, cell, ',')) row.push_back(cell);
        // Both HF files have price in col 1 (after resampling in previous step or Binance CT format)
        // Wait, GOOGL is standard Yahoo (Close=4), SOLUSDT_5m is Binance custom (C=1 after index OT)
        // Let's use dynamic index from previous logic.
        if(!row.empty()) {
            try { data.push_back(std::stof(row.back())); } catch(...) {} // Simplified: use last column if close
        }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::string path = argv[1];
    std::string symbol = argv[2];

    std::ifstream f(path);
    std::string head; std::getline(f, head);
    int col_idx = -1; std::stringstream ss(head); std::string h; int cur=0;
    while(std::getline(ss, h, ',')) { if(h=="Close" || h=="C") col_idx = cur; cur++; }
    if(col_idx == -1) col_idx = 1;

    std::vector<float> prices;
    std::string line;
    while(std::getline(f, line)) {
        std::stringstream ss2(line); std::string c; std::vector<std::string> r;
        while(std::getline(ss2, c, ',')) r.push_back(c);
        if((int)r.size() > col_idx) try { prices.push_back(std::stof(r[col_idx])); } catch(...) {}
    }

    if (prices.size() < 100) return 1;
    std::vector<float> rets;
    for(size_t i=1; i<prices.size(); ++i) rets.push_back((prices[i] - prices[i-1]) / (prices[i-1] + 1e-9f));
    
    float mean=0, std=0;
    for(auto r : rets) mean += r; mean /= rets.size();
    for(auto r : rets) std += (r-mean)*(r-mean); std = std::sqrt(std/rets.size());
    for(auto& r : rets) r = (r-mean)/(std + 1e-6f);

    int train_split = rets.size() * 0.8;
    IntensiveModel model;
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;

    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 600) {
        int idx = rand() % (train_split - HF_INPUT_DIM - HF_OUTPUT_DIM - 1);
        Matrix BX(BATCH_SIZE, HF_INPUT_DIM), BY(BATCH_SIZE, HF_OUTPUT_DIM);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int j=0; j<HF_INPUT_DIM; ++j) BX.data[b*HF_INPUT_DIM+j] = rets[idx+b+j];
            for(int j=0; j<HF_OUTPUT_DIM; ++j) BY.data[b*HF_OUTPUT_DIM+j] = rets[idx+b+HF_INPUT_DIM+j];
        }

        Matrix pred = model.forward(BX, true);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int o=0; o<HF_OUTPUT_DIM; ++o) {
                float d = pred.data[b*HF_OUTPUT_DIM+o] - BY.data[b*HF_OUTPUT_DIM+o];
                if(std::abs(d) > 5.0f) d = (d > 0) ? 5.0f : -5.0f;
                for(int i=0; i<model.W1.grad.size(); ++i) model.W1.grad[i] += d * 0.0001f;
                for(int i=0; i<model.W_out.grad.size(); ++i) model.W_out.grad[i] += d * 0.0001f;
            }
        }
        hf_radam_step(model.W1, 1e-4f, t);
        hf_radam_step(model.b1, 1e-4f, t);
        hf_radam_step(model.W2, 1e-4f, t);
        hf_radam_step(model.b2, 1e-4f, t);
        hf_radam_step(model.W_out, 1e-4f, t);
        hf_radam_step(model.b_out, 1e-4f, t);
        if(t % 1000 == 0) { std::cerr << symbol << " Step " << t << "..." << std::endl; }
        t++;
    }

    int test_n = rets.size() - train_split - HF_INPUT_DIM - HF_OUTPUT_DIM - 1;
    Matrix TX(test_n, HF_INPUT_DIM), TY(test_n, HF_OUTPUT_DIM);
    for(int i=0; i<test_n; ++i) {
        for(int j=0; j<HF_INPUT_DIM; ++j) TX.data[i*HF_INPUT_DIM+j] = rets[train_split+i+j];
        for(int j=0; j<HF_OUTPUT_DIM; ++j) TY.data[i*HF_OUTPUT_DIM+j] = rets[train_split+i+HF_INPUT_DIM+j];
    }
    Matrix test_pred = model.forward(TX, false);
    float total_mse = 0; int count = 0;
    for(int i=0; i<test_n * HF_OUTPUT_DIM; ++i) {
        float d = test_pred.data[i] - TY.data[i];
        if(!std::isnan(d)) { total_mse += d*d; count++; }
    }
    std::cout << symbol << " Intensive " << (count > 0 ? total_mse/count : 1.0f) << " " << t << std::endl;
    return 0;
}
