#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 64;  // 16 steps * 4 features
const int HIDDEN_DIM = 32; 
const int OUTPUT_DIM = 16; // 4 steps * 4 features
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

// Discontinuous PWL Model: Jump in value and derivative
struct JumpModel {
    int n_jumps;
    Matrix W1, W_out, slopes, intercepts;
    std::vector<float> knots;

    JumpModel(int n) : n_jumps(n), 
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), 
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        slopes(Matrix::random(HIDDEN_DIM, n)),
        intercepts(Matrix::random(HIDDEN_DIM, n)) {
        for (int i = 0; i <= n; ++i) knots.push_back(-4.0f + 8.0f * i / n);
    }

    Matrix forward(const Matrix& X) {
        Matrix h_lin(X.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X.rows, HIDDEN_DIM, INPUT_DIM, 1.0f, X.data.data(), INPUT_DIM, W1.data.data(), HIDDEN_DIM, 0.0f, h_lin.data.data(), HIDDEN_DIM);
        
        Matrix h_act(X.rows, HIDDEN_DIM);
        for (int b = 0; b < X.rows; ++b) {
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                float x = h_lin.data[b * HIDDEN_DIM + h];
                float y = 0;
                for (int i = 0; i < n_jumps; ++i) {
                    if (x > knots[i]) {
                        // Jump in value (intercept) + Jump in derivative (slope)
                        y += intercepts.data[h * n_jumps + i] + slopes.data[h * n_jumps + i] * (x - knots[i]);
                    }
                }
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

void benchmark(const std::vector<float>& X_train, const std::vector<float>& y_train, int n_jumps, int duration) {
    int n_samples = X_train.size() / INPUT_DIM;
    JumpModel model(n_jumps);
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    float best_loss = 1e18;

    while (true) {
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() >= duration) break;
        int idx = rand() % (n_samples - BATCH_SIZE);
        Matrix X(BATCH_SIZE, INPUT_DIM), Y(BATCH_SIZE, OUTPUT_DIM);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            for (int j = 0; j < INPUT_DIM; ++j) X.data[i * INPUT_DIM + j] = X_train[(idx + i) * INPUT_DIM + j];
            for (int j = 0; j < OUTPUT_DIM; ++j) Y.data[i * OUTPUT_DIM + j] = y_train[(idx + i) * OUTPUT_DIM + j];
        }
        Matrix pred = model.forward(X);
        float loss = 0;
        for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; ++i) {
            float d = pred.data[i] - Y.data[i];
            loss += d * d;
            model.W1.grad[0] += d * 0.0001f;
        }
        loss /= (BATCH_SIZE * OUTPUT_DIM);
        if (loss < best_loss) best_loss = loss;
        radam_step(model.W1, LEARNING_RATE, t);
        radam_step(model.W_out, LEARNING_RATE, t);
        radam_step(model.slopes, LEARNING_RATE, t);
        radam_step(model.intercepts, LEARNING_RATE, t);
        t++;
    }
    std::cout << "N=" << n_jumps << " Best_Loss=" << best_loss << " Steps=" << t << std::endl;
}

int main(int argc, char** argv) {
    auto X_train = load_npy("data/multistep/X_train.npy");
    auto y_train = load_npy("data/multistep/y_train.npy");
    
    if (argc > 1) {
        int n = std::stoi(argv[1]);
        int d = std::stoi(argv[2]);
        benchmark(X_train, y_train, n, d);
        return 0;
    }

    std::vector<int> ns = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    for (int n : ns) {
        benchmark(X_train, y_train, n, 60);
    }
    return 0;
}
