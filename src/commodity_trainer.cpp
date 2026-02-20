#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_STEPS = 16;
const int HIDDEN_DIM = 32;
const int OUTPUT_DIM = 1;
const int BATCH_SIZE = 32;
const float CLIP_VAL = 5.0f;

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
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (auto& val : mat.data) val = dist(gen);
        return mat;
    }

    static Matrix zeros(int r, int c) {
        Matrix mat(r, c);
        std::fill(mat.data.begin(), mat.data.end(), 0.0f);
        return mat;
    }
};

void radam_step(Matrix& p, float lr, int t) {
    float norm = 0;
    vDSP_dotpr(p.grad.data(), 1, p.grad.data(), 1, &norm, p.grad.size());
    norm = std::sqrt(norm);
    if (norm > CLIP_VAL) {
        float scale = CLIP_VAL / norm;
        vDSP_vsmul(p.grad.data(), 1, &scale, p.grad.data(), 1, p.grad.size());
    }
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for (size_t i = 0; i < p.data.size(); ++i) {
        p.m[i] = beta1 * p.m[i] + (1.0f - beta1) * p.grad[i];
        p.v[i] = beta2 * p.v[i] + (1.0f - beta2) * p.grad[i] * p.grad[i];
        float m_hat = p.m[i] / (1.0f - std::pow(beta1, t));
        float v_hat = std::sqrt(p.v[i] / (1.0f - std::pow(beta2, t)));
        p.data[i] -= lr * m_hat / (v_hat + eps);
        p.grad[i] = 0;
    }
}

struct FFNN32 {
    Matrix W1, b1, W_out, b_out;
    FFNN32() : 
        W1(Matrix::random(INPUT_STEPS, HIDDEN_DIM)), 
        b1(Matrix::zeros(1, HIDDEN_DIM)),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        b_out(Matrix::zeros(1, OUTPUT_DIM)) {}

    Matrix forward(const Matrix& x) {
        Matrix h(x.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, INPUT_STEPS, 1.0f, x.data.data(), INPUT_STEPS, W1.data.data(), HIDDEN_DIM, 0.0f, h.data.data(), HIDDEN_DIM);
        for(int i=0; i<h.rows; ++i) vDSP_vadd(h.data.data() + i*HIDDEN_DIM, 1, b1.data.data(), 1, h.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h.data) v = std::tanh(v);
        Matrix y(x.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, OUTPUT_DIM, HIDDEN_DIM, 1.0f, h.data.data(), HIDDEN_DIM, W_out.data.data(), OUTPUT_DIM, 0.0f, y.data.data(), OUTPUT_DIM);
        for(int i=0; i<y.rows; ++i) vDSP_vadd(y.data.data() + i*OUTPUT_DIM, 1, b_out.data.data(), 1, y.data.data() + i*OUTPUT_DIM, 1, OUTPUT_DIM);
        return y;
    }
};

std::vector<float> load_column(int col_idx) {
    std::vector<float> data;
    std::ifstream file("data/commodities/returns_15m.csv");
    std::string line; std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); 
        for(int i=0; i<=col_idx; ++i) std::getline(ss, cell, ',');
        try { data.push_back(std::stof(cell)); } catch(...) { data.push_back(0.0f); }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 5) return 1;
    int col_idx = std::stoi(argv[1]);
    float lr = std::stof(argv[2]);
    float threshold = std::stof(argv[3]);
    int seed = std::stoi(argv[4]);
    
    std::srand(seed);
    auto returns = load_column(col_idx);
    int n_samples = returns.size();
    int train_split = n_samples * 0.8;

    FFNN32 model;
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 60) {
        int idx = rand() % (train_split - INPUT_STEPS - 1);
        Matrix BX(BATCH_SIZE, INPUT_STEPS), BY(BATCH_SIZE, OUTPUT_DIM);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int j=0; j<INPUT_STEPS; ++j) BX.data[b*INPUT_STEPS+j] = returns[idx+b+j];
            BY.data[b] = returns[idx+b+INPUT_STEPS];
        }
        Matrix pred = model.forward(BX);
        for(int b=0; b<BATCH_SIZE; ++b) {
            float d = pred.data[b] - BY.data[b];
            model.W1.grad[0] += d * 0.0001f;
        }
        radam_step(model.W1, lr, t);
        radam_step(model.W_out, lr, t);
        t++;
    }

    float capital = 100.0f;
    for (int i = train_split; i < n_samples - 1; ++i) {
        Matrix TX(1, INPUT_STEPS);
        for(int j=0; j<INPUT_STEPS; ++j) TX.data[j] = returns[i - INPUT_STEPS + j];
        Matrix pred = model.forward(TX);
        float p = pred.data[0];
        float actual = returns[i];
        
        if (p > threshold) capital += capital * actual - 0.01f; // Smaller commission for retail/fx
        else if (p < -threshold) capital -= capital * actual + 0.01f;
        if (capital <= 0) { capital = 0; break; }
    }

    std::cout << capital << std::endl;
    return 0;
}
