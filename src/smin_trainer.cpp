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
const int PWL_PIECES = 128;
const int BATCH_SIZE = 32;
const float LEARNING_RATE = 0.001f;
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

struct PWLModel {
    Matrix W1, W_out, slopes;
    std::vector<float> breakpoints;

    PWLModel() : 
        W1(Matrix::random(INPUT_STEPS, HIDDEN_DIM)), 
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        slopes(Matrix::random(HIDDEN_DIM, PWL_PIECES)) {
        for (int i = 0; i <= PWL_PIECES; ++i) breakpoints.push_back(-0.05f + 0.1f * i / PWL_PIECES);
    }

    Matrix forward(const Matrix& X) {
        Matrix h_lin(X.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X.rows, HIDDEN_DIM, INPUT_STEPS, 1.0f, X.data.data(), INPUT_STEPS, W1.data.data(), HIDDEN_DIM, 0.0f, h_lin.data.data(), HIDDEN_DIM);
        Matrix h_act(X.rows, HIDDEN_DIM);
        for (int b = 0; b < X.rows; ++b) {
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                float x = h_lin.data[b * HIDDEN_DIM + h];
                float y = 0;
                for (int i = 0; i < PWL_PIECES; ++i) y += slopes.data[h * PWL_PIECES + i] * std::max(0.0f, x - breakpoints[i]);
                h_act.data[b * HIDDEN_DIM + h] = y;
            }
        }
        Matrix pred(X.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X.rows, OUTPUT_DIM, HIDDEN_DIM, 1.0f, h_act.data.data(), HIDDEN_DIM, W_out.data.data(), OUTPUT_DIM, 0.0f, pred.data.data(), OUTPUT_DIM);
        return pred;
    }
};

std::vector<float> load_data() {
    std::vector<float> data;
    std::ifstream file("data/india_etf/smin_15m.csv");
    std::string line; std::getline(file, line); // header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        // Format: Date,Open,High,Low,Close,Adj Close,Volume,Log_Return
        // We want Log_Return (last column usually, but let's parse carefuly)
        std::vector<std::string> row;
        while(std::getline(ss, cell, ',')) row.push_back(cell);
        if(!row.empty()) {
            try { data.push_back(std::stof(row.back())); } catch(...) { data.push_back(0.0f); }
        }
    }
    return data;
}

int main() {
    auto returns = load_data();
    int n_samples = returns.size();
    if(n_samples < 100) return 1;
    
    int train_split = n_samples * 0.8;
    PWLModel model;
    
    // Train for 60s
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    while(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 60) {
        int idx = rand() % (train_split - INPUT_STEPS - 1);
        Matrix BX(BATCH_SIZE, INPUT_STEPS), BY(BATCH_SIZE, OUTPUT_DIM);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int j=0; j<INPUT_STEPS; ++j) BX.data[b*INPUT_STEPS+j] = returns[idx+b+j];
            BY.data[b] = returns[idx+b+INPUT_STEPS];
        }
        Matrix pred = model.forward(BX);
        for(int b=0; b<BATCH_SIZE; ++b) {
            float d = pred.data[b] - BY.data[b];
            model.W1.grad[0] += d * 0.001f;
        }
        radam_step(model.W1, LEARNING_RATE, t);
        radam_step(model.W_out, LEARNING_RATE, t);
        radam_step(model.slopes, LEARNING_RATE, t);
        t++;
    }
    
    // Simulation
    float capital = 100.0f;
    for(int i = train_split; i < n_samples - INPUT_STEPS; ++i) {
        Matrix TX(1, INPUT_STEPS);
        for(int j=0; j<INPUT_STEPS; ++j) TX.data[j] = returns[i+j];
        Matrix pred = model.forward(TX);
        float p = pred.data[0];
        float actual = returns[i+INPUT_STEPS];
        
        // Simple strategy
        if(p > 0.0005f) capital += capital * actual - 0.1f; // Commission
        else if(p < -0.0005f) capital -= capital * actual + 0.1f;
        if(capital <= 0) { capital = 0; break; }
    }
    
    std::cout << capital << std::endl;
    return 0;
}
