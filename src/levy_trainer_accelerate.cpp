#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <Accelerate/Accelerate.h>
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

void radam_step(Matrix& p, float lr, int t) {
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

struct Model {
    Matrix W1, b1, W2, b2, W_out, b_out;
    std::mt19937 gen;

    Model() : 
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), b1(HIDDEN_DIM, 1),
        W2(Matrix::random(HIDDEN_DIM, HIDDEN_DIM)), b2(HIDDEN_DIM, 1),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)), b_out(OUTPUT_DIM, 1) {
        std::random_device rd;
        gen.seed(rd());
    }

    Matrix forward(const Matrix& x, bool training) {
        Matrix h1(x.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, INPUT_DIM, 1.0f, x.data.data(), INPUT_DIM, W1.data.data(), HIDDEN_DIM, 0.0f, h1.data.data(), HIDDEN_DIM);
        for(int i=0; i<h1.rows; ++i) vDSP_vadd(h1.data.data() + i*HIDDEN_DIM, 1, b1.data.data(), 1, h1.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h1.data) {
            v = std::tanh(v);
            if(training) v += generate_stable(gen);
        }

        Matrix h2(x.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, HIDDEN_DIM, 1.0f, h1.data.data(), HIDDEN_DIM, W2.data.data(), HIDDEN_DIM, 0.0f, h2.data.data(), HIDDEN_DIM);
        for(int i=0; i<h2.rows; ++i) vDSP_vadd(h2.data.data() + i*HIDDEN_DIM, 1, b2.data.data(), 1, h2.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h2.data) {
            v = std::tanh(v);
            if(training) v += generate_stable(gen);
        }

        Matrix y(x.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, OUTPUT_DIM, HIDDEN_DIM, 1.0f, h2.data.data(), HIDDEN_DIM, W_out.data.data(), OUTPUT_DIM, 0.0f, y.data.data(), OUTPUT_DIM);
        for(int i=0; i<y.rows; ++i) vDSP_vadd(y.data.data() + i*OUTPUT_DIM, 1, b_out.data.data(), 1, y.data.data() + i*OUTPUT_DIM, 1, OUTPUT_DIM);
        return y;
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
        int idx = rand() % (train_split - INPUT_DIM - 1);
        Matrix BX(BATCH_SIZE, INPUT_DIM), BY(BATCH_SIZE, OUTPUT_DIM);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int j=0; j<INPUT_DIM; ++j) BX.data[b*INPUT_DIM+j] = rets[idx+b+j];
            BY.data[b] = rets[idx+b+INPUT_DIM];
        }
        Matrix pred = model.forward(BX, true);
        for(int b=0; b<BATCH_SIZE; ++b) {
            float d = pred.data[b] - BY.data[b];
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
    Matrix TX(test_n, INPUT_DIM), TY(test_n, OUTPUT_DIM);
    for(int i=0; i<test_n; ++i) {
        for(int j=0; j<INPUT_DIM; ++j) TX.data[i*INPUT_DIM+j] = rets[train_split+i+j];
        TY.data[i] = rets[train_split+i+INPUT_DIM];
    }
    Matrix test_pred = model.forward(TX, false);
    float total_mse = 0; int count = 0;
    for(int i=0; i<test_n; ++i) {
        float d = test_pred.data[i] - TY.data[i];
        if(!std::isnan(d)) { total_mse += d*d; count++; }
    }
    std::cout << symbol << " Accelerate " << (count > 0 ? total_mse/count : 1.0f) << " " << t << std::endl;
    return 0;
}
