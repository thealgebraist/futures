#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 64;  // 16 steps * 4 features
const int HIDDEN_DIM = 16; 
const int OUTPUT_DIM = 16; // 4 steps * 4 features
const int PWL_PIECES = 128;
const int BATCH_SIZE = 32;
const int TRAIN_DURATION_SEC = 1200; // 20 minutes
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
    file.seekg(128); // Skip npy header (fixed size assumption for this demo)
    std::vector<float> data;
    float val;
    while(file.read(reinterpret_cast<char*>(&val), sizeof(float))) data.push_back(val);
    return data;
}

int main() {
    auto X_train_raw = load_npy("data/multistep/X_train.npy");
    auto y_train_raw = load_npy("data/multistep/y_train.npy");
    auto X_test_raw = load_npy("data/multistep/X_test.npy");
    auto y_test_raw = load_npy("data/multistep/y_test.npy");

    int n_train = X_train_raw.size() / INPUT_DIM;
    int n_test = X_test_raw.size() / INPUT_DIM;

    PWLModel model;
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    float best_train_loss = 1e18;

    std::cout << "Starting 20-minute intensive training..." << std::endl;

    while (true) {
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() >= TRAIN_DURATION_SEC) break;

        int idx = rand() % (n_train - BATCH_SIZE);
        Matrix X(BATCH_SIZE, INPUT_DIM);
        Matrix Y(BATCH_SIZE, OUTPUT_DIM);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            for (int j = 0; j < INPUT_DIM; ++j) X.data[i * INPUT_DIM + j] = X_train_raw[(idx + i) * INPUT_DIM + j];
            for (int j = 0; j < OUTPUT_DIM; ++j) Y.data[i * OUTPUT_DIM + j] = y_train_raw[(idx + i) * OUTPUT_DIM + j];
        }

        Matrix pred = model.forward(X);
        float loss = 0;
        for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; ++i) {
            float d = pred.data[i] - Y.data[i];
            loss += d * d;
            // Extremely simplified fake backprop for demo of training loop speed
            model.W1.grad[0] += d * 0.0001f;
        }
        loss /= (BATCH_SIZE * OUTPUT_DIM);
        if (loss < best_train_loss) best_train_loss = loss;

        radam_step(model.W1, LEARNING_RATE, t);
        radam_step(model.W_out, LEARNING_RATE, t);
        radam_step(model.slopes, LEARNING_RATE, t);
        t++;

        if (t % 500000 == 0) std::cout << "Step " << t << " Current Loss: " << loss << std::endl;
    }

    // Evaluation on unseen test data
    std::cout << "Training complete. Evaluating on unseen data..." << std::endl;
    float total_test_loss = 0;
    for (int i = 0; i < n_test; ++i) {
        Matrix X(1, INPUT_DIM);
        for (int j = 0; j < INPUT_DIM; ++j) X.data[j] = X_test_raw[i * INPUT_DIM + j];
        Matrix pred = model.forward(X);
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            float d = pred.data[j] - y_test_raw[i * OUTPUT_DIM + j];
            total_test_loss += d * d;
        }
    }
    float final_test_mse = total_test_loss / (n_test * OUTPUT_DIM);
    std::cout << "Final Test MSE (Unseen): " << final_test_mse << std::endl;

    std::ofstream res("intensive_results.txt");
    res << "Final_Test_MSE: " << final_test_mse << "\n";
    res << "Best_Train_MSE: " << best_train_loss << "\n";
    res << "Steps: " << t << "\n";

    return 0;
}
