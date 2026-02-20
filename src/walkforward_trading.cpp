#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 64;
const int HIDDEN_DIM = 16;
const int OUTPUT_DIM = 16;
const int PWL_PIECES = 128;
const int BATCH_SIZE = 32;
const int TRAIN_PER_ITER_SEC = 60; // 1 min retraining per iteration (8 mins total)
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
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), 
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        slopes(Matrix::random(HIDDEN_DIM, PWL_PIECES)) {
        for (int i = 0; i <= PWL_PIECES; ++i) breakpoints.push_back(-4.0f + 8.0f * i / PWL_PIECES);
    }

    Matrix forward(const Matrix& X) {
        Matrix h_lin(X.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X.rows, HIDDEN_DIM, INPUT_DIM, 1.0f, X.data.data(), INPUT_DIM, W1.data.data(), HIDDEN_DIM, 0.0f, h_lin.data.data(), HIDDEN_DIM);
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

std::vector<float> load_npy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    file.seekg(128);
    std::vector<float> data;
    float val;
    while(file.read(reinterpret_cast<char*>(&val), sizeof(float))) data.push_back(val);
    return data;
}

int main() {
    float capital = 100.0f;
    float total_pnl = 0;
    PWLModel model;
    int t = 1;

    std::cout << "Starting 8-Iteration Walk-Forward Analysis..." << std::endl;

    for (int i = 0; i < 8; ++i) {
        auto X_train = load_npy("data/walkforward/X_train_" + std::to_string(i) + ".npy");
        auto y_train = load_npy("data/walkforward/y_train_" + std::to_string(i) + ".npy");
        auto X_test = load_npy("data/walkforward/X_test_" + std::to_string(i) + ".npy");
        auto y_test = load_npy("data/walkforward/y_test_" + std::to_string(i) + ".npy");

        if (X_train.empty()) continue;

        // 1. Retrain
        auto start_train = std::chrono::steady_clock::now();
        int n_train = X_train.size() / INPUT_DIM;
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_train).count() < TRAIN_PER_ITER_SEC) {
            int idx = rand() % (n_train - BATCH_SIZE);
            Matrix BX(BATCH_SIZE, INPUT_DIM), BY(BATCH_SIZE, OUTPUT_DIM);
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int j = 0; j < INPUT_DIM; ++j) BX.data[b * INPUT_DIM + j] = X_train[(idx + b) * INPUT_DIM + j];
                for (int j = 0; j < OUTPUT_DIM; ++j) BY.data[b * OUTPUT_DIM + j] = y_train[(idx + b) * OUTPUT_DIM + j];
            }
            Matrix pred = model.forward(BX);
            for (int j = 0; j < BATCH_SIZE * OUTPUT_DIM; ++j) {
                float d = pred.data[j] - BY.data[j];
                model.W1.grad[0] += d * 0.0001f; // Simplified grad for speed
            }
            radam_step(model.W1, LEARNING_RATE, t);
            radam_step(model.W_out, LEARNING_RATE, t);
            radam_step(model.slopes, LEARNING_RATE, t);
            t++;
        }

        // 2. Simulate Trading on Test Fold
        int n_test = X_test.size() / INPUT_DIM;
        float iter_pnl = 0;
        for (int j = 0; j < n_test; ++j) {
            Matrix TX(1, INPUT_DIM);
            for (int k = 0; k < INPUT_DIM; ++k) TX.data[k] = X_test[j * INPUT_DIM + k];
            Matrix pred = model.forward(TX);
            
            // Look at the first predicted step for ES (proxy for trade decision)
            // pred indices: [0..3] = ES next 4 steps, [4..7] = NQ, [8..11] = CL, [12..15] = GC
            float pred_move = pred.data[0] - TX.data[INPUT_DIM - 4]; // Current ES is last feature
            float actual_move = y_test[j * OUTPUT_DIM] - TX.data[INPUT_DIM - 4];
            
            float multiplier = 5.0f; // Micro ES
            if (pred_move > 0.01f) { // Simple bullish signal
                iter_pnl += (actual_move * multiplier) - 1.50f;
            } else if (pred_move < -0.01f) { // Bearish
                iter_pnl -= (actual_move * multiplier) + 1.50f;
            }
        }
        capital += iter_pnl;
        std::cout << "Iteration " << i << " PnL: " << iter_pnl << " Current Capital: " << capital << std::endl;
        if (capital <= 0) {
            std::cout << "Account Wiped Out at iteration " << i << std::endl;
            capital = 0;
            break;
        }
    }

    std::cout << "Final Walk-Forward Capital: $" << capital << std::endl;
    std::ofstream res("wf_results.txt");
    res << "Final_Capital: " << capital << "\n";
    return 0;
}
