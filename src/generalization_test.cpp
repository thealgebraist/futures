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
const int N_FUTURES = 16;
const int INPUT_DIM = INPUT_STEPS * N_FUTURES;
const int OUTPUT_DIM = N_FUTURES;
const int PWL_PIECES = 128;
const int BATCH_SIZE = 32;
const int TRAIN_TOTAL_SEC = 600; // 10 minutes total
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
    int hidden_dim;
    Matrix W1, W_out, slopes;
    std::vector<float> breakpoints;

    PWLModel(int h) : hidden_dim(h), 
        W1(Matrix::random(INPUT_DIM, h)), 
        W_out(Matrix::random(h, OUTPUT_DIM)),
        slopes(Matrix::random(h, PWL_PIECES)) {
        for (int i = 0; i <= PWL_PIECES; ++i) breakpoints.push_back(-0.1f + 0.2f * i / PWL_PIECES); // Return-scale knots
    }

    Matrix forward(const Matrix& X) {
        Matrix h_lin(X.rows, hidden_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X.rows, hidden_dim, INPUT_DIM, 1.0f, X.data.data(), INPUT_DIM, W1.data.data(), hidden_dim, 0.0f, h_lin.data.data(), hidden_dim);
        Matrix h_act(X.rows, hidden_dim);
        for (int b = 0; b < X.rows; ++b) {
            for (int h = 0; h < hidden_dim; ++h) {
                float x = h_lin.data[b * hidden_dim + h];
                float y = 0;
                for (int i = 0; i < PWL_PIECES; ++i) y += slopes.data[h * PWL_PIECES + i] * std::max(0.0f, x - breakpoints[i]);
                h_act.data[b * hidden_dim + h] = y;
            }
        }
        Matrix pred(X.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X.rows, OUTPUT_DIM, hidden_dim, 1.0f, h_act.data.data(), hidden_dim, W_out.data.data(), OUTPUT_DIM, 0.0f, pred.data.data(), OUTPUT_DIM);
        return pred;
    }
};

std::vector<std::vector<float>> load_csv() {
    std::vector<std::vector<float>> data;
    std::ifstream file("data/futures_16_changes.csv");
    std::string line; std::getline(file, line); // header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell; std::vector<float> row;
        std::getline(ss, cell, ','); // skip index
        while (std::getline(ss, cell, ',')) {
            try { row.push_back(std::stof(cell)); } catch (...) { row.push_back(0.0f); }
        }
        if (row.size() == N_FUTURES) data.push_back(row);
    }
    return data;
}

int main() {
    auto raw_data = load_csv();
    int n_samples = raw_data.size();
    
    // Prepare sliding windows
    std::vector<std::vector<float>> X, Y;
    for (int i = 0; i < n_samples - INPUT_STEPS - 1; ++i) {
        std::vector<float> x_win;
        for (int s = 0; i+s < i + INPUT_STEPS; ++s) {
            for (float val : raw_data[i+s]) x_win.push_back(val);
        }
        X.push_back(x_win);
        Y.push_back(raw_data[i + INPUT_STEPS]);
    }

    int total_wins = X.size();
    int train_split = total_wins * 0.8;
    
    std::vector<int> neuron_counts = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int time_per_model = TRAIN_TOTAL_SEC / neuron_counts.size();

    std::cout << "Starting Generalization Scaling Test..." << std::endl;
    std::ofstream res_file("gen_test_results.txt");

    for (int n_neurons : neuron_counts) {
        PWLModel model(n_neurons);
        auto start = std::chrono::steady_clock::now();
        int t = 1;
        
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < time_per_model) {
            int idx = rand() % (train_split - BATCH_SIZE);
            Matrix BX(BATCH_SIZE, INPUT_DIM), BY(BATCH_SIZE, OUTPUT_DIM);
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int j = 0; j < INPUT_DIM; ++j) BX.data[b * INPUT_DIM + j] = X[idx + b][j];
                for (int j = 0; j < OUTPUT_DIM; ++j) BY.data[b * OUTPUT_DIM + j] = Y[idx + b][j];
            }
            Matrix pred = model.forward(BX);
            for (int j = 0; j < BATCH_SIZE * OUTPUT_DIM; ++j) {
                float d = pred.data[j] - BY.data[j];
                model.W1.grad[0] += d * 0.0001f;
            }
            radam_step(model.W1, LEARNING_RATE, t);
            radam_step(model.W_out, LEARNING_RATE, t);
            radam_step(model.slopes, LEARNING_RATE, t);
            t++;
        }

        // Test on unseen data
        float total_test_mse = 0;
        int test_count = 0;
        for (int i = train_split; i < total_wins; ++i) {
            Matrix TX(1, INPUT_DIM);
            for (int j = 0; j < INPUT_DIM; ++j) TX.data[j] = X[i][j];
            Matrix pred = model.forward(TX);
            for (int j = 0; j < OUTPUT_DIM; ++j) {
                float d = pred.data[j] - Y[i][j];
                total_test_mse += d * d;
            }
            test_count++;
        }
        float final_mse = total_test_mse / (test_count * OUTPUT_DIM);
        std::cout << "Neurons=" << n_neurons << " Test_MSE=" << final_mse << std::endl;
        res_file << n_neurons << " " << final_mse << "\n";
    }

    return 0;
}
