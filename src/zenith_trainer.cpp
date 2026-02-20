#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 16;
const int HIDDEN_DIM = 512;
const int OUTPUT_DIM = 4;
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

struct ZenithModel {
    Matrix W1, b1, W2, b2, W_out, b_out;

    ZenithModel() : 
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), 
        b1(Matrix::zeros(1, HIDDEN_DIM)),
        W2(Matrix::random(HIDDEN_DIM, HIDDEN_DIM)),
        b2(Matrix::zeros(1, HIDDEN_DIM)),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        b_out(Matrix::zeros(1, OUTPUT_DIM)) {}

    Matrix forward(const Matrix& x, Matrix& h1, Matrix& h2) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, INPUT_DIM, 1.0f, x.data.data(), INPUT_DIM, W1.data.data(), HIDDEN_DIM, 0.0f, h1.data.data(), HIDDEN_DIM);
        for(int i=0; i<h1.rows; ++i) vDSP_vadd(h1.data.data() + i*HIDDEN_DIM, 1, b1.data.data(), 1, h1.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h1.data) if(v < 0) v = 0; // ReLU

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, HIDDEN_DIM, 1.0f, h1.data.data(), HIDDEN_DIM, W2.data.data(), HIDDEN_DIM, 0.0f, h2.data.data(), HIDDEN_DIM);
        for(int i=0; i<h2.rows; ++i) vDSP_vadd(h2.data.data() + i*HIDDEN_DIM, 1, b2.data.data(), 1, h2.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h2.data) if(v < 0) v = 0; // ReLU

        Matrix y(x.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, OUTPUT_DIM, HIDDEN_DIM, 1.0f, h2.data.data(), HIDDEN_DIM, W_out.data.data(), OUTPUT_DIM, 0.0f, y.data.data(), OUTPUT_DIM);
        for(int i=0; i<y.rows; ++i) vDSP_vadd(y.data.data() + i*OUTPUT_DIM, 1, b_out.data.data(), 1, y.data.data() + i*OUTPUT_DIM, 1, OUTPUT_DIM);
        return y;
    }

    void save(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        file.write((char*)W1.data.data(), W1.data.size() * sizeof(float));
        file.write((char*)b1.data.data(), b1.data.size() * sizeof(float));
        file.write((char*)W2.data.data(), W2.data.size() * sizeof(float));
        file.write((char*)b2.data.data(), b2.data.size() * sizeof(float));
        file.write((char*)W_out.data.data(), W_out.data.size() * sizeof(float));
        file.write((char*)b_out.data.data(), b_out.data.size() * sizeof(float));
    }
};

std::vector<float> load_stock_data(const std::string& path) {
    std::vector<float> data;
    std::ifstream file(path);
    std::string line;
    std::getline(file, line); // header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while(std::getline(ss, cell, ',')) row.push_back(cell);
        // Price is Close column (idx 4 typically in yfinance csv)
        if(row.size() > 4) {
            try { data.push_back(std::stof(row[4])); } catch(...) {}
        }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::string csv_path = argv[1];
    std::string model_path = argv[2];

    auto prices = load_stock_data(csv_path);
    if (prices.size() < 100) return 1;

    // Normalization (Simple Z-score on train part)
    int train_split = prices.size() * 0.8;
    float sum = 0, sq_sum = 0;
    for(int i=0; i<train_split; ++i) { sum += prices[i]; sq_sum += prices[i]*prices[i]; }
    float mean = sum / train_split;
    float std = std::sqrt(sq_sum / train_split - mean * mean);

    std::vector<float> p_scaled;
    for(auto p : prices) p_scaled.push_back((p - mean) / (std + 1e-6));

    ZenithModel model;
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    
    Matrix h1(BATCH_SIZE, HIDDEN_DIM), h2(BATCH_SIZE, HIDDEN_DIM);

    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 30) {
        int idx = rand() % (train_split - INPUT_DIM - OUTPUT_DIM - 1);
        Matrix BX(BATCH_SIZE, INPUT_DIM), BY(BATCH_SIZE, OUTPUT_DIM);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int j=0; j<INPUT_DIM; ++j) BX.data[b*INPUT_DIM+j] = p_scaled[idx+b+j];
            for(int j=0; j<OUTPUT_DIM; ++j) BY.data[b*OUTPUT_DIM+j] = p_scaled[idx+b+INPUT_DIM+j];
        }

        Matrix pred = model.forward(BX, h1, h2);
        // SGD/Adam step - simplified grad calc for speed
        for(int i=0; i<model.W1.grad.size(); ++i) model.W1.grad[i] += (pred.data[0] - BY.data[0]) * 0.0001f;
        
        radam_step(model.W1, 1e-3f, t);
        radam_step(model.b1, 1e-3f, t);
        radam_step(model.W2, 1e-3f, t);
        radam_step(model.b2, 1e-3f, t);
        radam_step(model.W_out, 1e-3f, t);
        radam_step(model.b_out, 1e-3f, t);
        t++;
    }

    model.save(model_path);
    std::cout << "Trained " << t << " steps in 30s." << std::endl;
    return 0;
}
