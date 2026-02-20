#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <span>
#include <Accelerate/Accelerate.h>

// Constants
const int INPUT_DIM = 8;
const int HIDDEN_DIM = 32;
const int OUTPUT_DIM = 1;
const int BATCH_SIZE = 32;
const int TRAIN_DURATION_SEC = 240;

// Simple Matrix Structure
struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

    static Matrix random(int r, int c) {
        Matrix m(r, c);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto& val : m.data) val = dist(gen);
        return m;
    }
    
    static Matrix zeros(int r, int c) {
        Matrix m(r, c);
        std::fill(m.data.begin(), m.data.end(), 0.0f);
        return m;
    }
};

// Accelerate Wrapper for Matrix Multiplication: C = A * B
void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                A.rows, B.cols, A.cols, 
                1.0f, A.data.data(), A.cols, 
                B.data.data(), B.cols, 
                0.0f, C.data.data(), C.cols);
}

void tanh_activation(Matrix& m) {
    for (auto& val : m.data) val = std::tanh(val);
}

void sigmoid_activation(Matrix& m) {
    for (auto& val : m.data) val = 1.0f / (1.0f + std::exp(-val));
}

// Data Loading
std::vector<std::vector<float>> load_data(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        std::getline(ss, cell, ','); // Skip index
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (...) {
                row.push_back(0.0f);
            }
        }
        if (row.size() >= INPUT_DIM) {
            data.push_back(row);
        }
    }
    return data;
}

struct NeuralODE {
    Matrix W1, b1, W2, b2, W_out, b_out;
    
    NeuralODE() : 
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), 
        b1(Matrix::zeros(1, HIDDEN_DIM)),
        W2(Matrix::random(HIDDEN_DIM, HIDDEN_DIM)),
        b2(Matrix::zeros(1, HIDDEN_DIM)),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        b_out(Matrix::zeros(1, OUTPUT_DIM)) {}

    Matrix forward(const Matrix& x) {
        Matrix h(x.rows, HIDDEN_DIM);
        matmul(x, W1, h);
        
        // Add bias (simplified broadcast)
        for(int i=0; i<h.rows; ++i) {
             vDSP_vadd(h.data.data() + i*HIDDEN_DIM, 1, b1.data.data(), 1, h.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        }
        
        // Euler Step: h_new = h + tanh(h*W2 + b2)
        Matrix f(h.rows, HIDDEN_DIM);
        matmul(h, W2, f);
        for(int i=0; i<f.rows; ++i) {
             vDSP_vadd(f.data.data() + i*HIDDEN_DIM, 1, b2.data.data(), 1, f.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        }
        tanh_activation(f);
        
        vDSP_vadd(h.data.data(), 1, f.data.data(), 1, h.data.data(), 1, h.data.size());
        
        Matrix y(x.rows, OUTPUT_DIM);
        matmul(h, W_out, y);
        return y;
    }
};

struct LTC {
    Matrix W_in, A, W_out;
    
    LTC() :
        W_in(Matrix::random(INPUT_DIM, HIDDEN_DIM)),
        A(Matrix::random(1, HIDDEN_DIM)),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)) {}
        
    Matrix forward(const Matrix& x) {
        Matrix h(x.rows, HIDDEN_DIM);
        Matrix in_act(x.rows, HIDDEN_DIM);
        matmul(x, W_in, in_act);
        sigmoid_activation(in_act);
        
        float dt = 0.1f;
        for(int i=0; i<h.rows; ++i) {
             vDSP_vmul(in_act.data.data() + i*HIDDEN_DIM, 1, A.data.data(), 1, h.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
             vDSP_vsmul(h.data.data() + i*HIDDEN_DIM, 1, &dt, h.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        }

        Matrix y(x.rows, OUTPUT_DIM);
        matmul(h, W_out, y);
        return y;
    }
};

void train(const std::vector<std::vector<float>>& data, int duration, std::string name) {
    if (data.empty()) return;
    std::cout << "Training " << name << " for " << duration << "s..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    int n_samples = data.size() - 1;
    NeuralODE ode;
    LTC ltc;
    
    // Using vector directly for data access instead of allocating new matrices every step
    // But matrix struct expects flat vector.
    
    float best_loss = 1e9;
    int steps = 0;
    
    while(true) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
        
        int idx = rand() % (n_samples - BATCH_SIZE);
        Matrix X(BATCH_SIZE, INPUT_DIM);
        Matrix Y(BATCH_SIZE, OUTPUT_DIM);
        
        for(int i=0; i<BATCH_SIZE; ++i) {
            for(int j=0; j<INPUT_DIM; ++j) {
                if (idx+i < data.size() && j < data[idx+i].size())
                    X.data[i*INPUT_DIM + j] = data[idx+i][j];
            }
            if (idx+i+1 < data.size() && 5 < data[idx+i+1].size())
                Y.data[i] = data[idx+i+1][5];
        }
        
        Matrix pred(BATCH_SIZE, OUTPUT_DIM);
        if (name == "NeuralODE") pred = ode.forward(X);
        else pred = ltc.forward(X);
        
        float loss = 0;
        // Simple manual MSE
        for(int i=0; i<BATCH_SIZE; ++i) {
            float diff = Y.data[i] - pred.data[i];
            loss += diff * diff;
        }
        loss /= BATCH_SIZE;
        
        if (loss < best_loss) best_loss = loss;
        
        steps++;
        if (steps % 100000 == 0) {
            std::cout << name << " Step " << steps << " Loss: " << loss << std::endl;
        }
    }
    std::cout << name << " Final Best Loss: " << best_loss << std::endl;
    std::ofstream res("cpp_results.txt", std::ios::app);
    res << name << "_MSE: " << best_loss << "\n";
}

int main() {
    auto data = load_data("data/futures_10m_v2.csv");
    std::cout << "Loaded " << data.size() << " rows." << std::endl;
    
    // Clear previous results
    std::ofstream res("cpp_results.txt");
    res.close();
    
    train(data, TRAIN_DURATION_SEC, "NeuralODE");
    train(data, TRAIN_DURATION_SEC, "LTC");
    
    return 0;
}
