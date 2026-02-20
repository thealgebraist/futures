#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 64; 
const int HIDDEN_DIM = 64;
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
        float m_hat = p.m[i] / (1.0f - std::pow(beta1, (float)t));
        float v_hat = std::sqrt(p.v[i] / (1.0f - std::pow(beta2, (float)t)));
        p.data[i] -= lr * m_hat / (v_hat + eps);
        p.grad[i] = 0;
    }
}

struct FFNN64_64 {
    Matrix W1, b1, W2, b2, W_out, b_out;
    FFNN64_64() : 
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), 
        b1(Matrix::zeros(1, HIDDEN_DIM)),
        W2(Matrix::random(HIDDEN_DIM, HIDDEN_DIM)),
        b2(Matrix::zeros(1, HIDDEN_DIM)),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        b_out(Matrix::zeros(1, OUTPUT_DIM)) {}

    Matrix forward(const Matrix& x) {
        Matrix h1(x.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, INPUT_DIM, 1.0f, x.data.data(), INPUT_DIM, W1.data.data(), HIDDEN_DIM, 0.0f, h1.data.data(), HIDDEN_DIM);
        for(int i=0; i<h1.rows; ++i) vDSP_vadd(h1.data.data() + i*HIDDEN_DIM, 1, b1.data.data(), 1, h1.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h1.data) v = std::tanh(v);

        Matrix h2(x.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, HIDDEN_DIM, 1.0f, h1.data.data(), HIDDEN_DIM, W2.data.data(), HIDDEN_DIM, 0.0f, h2.data.data(), HIDDEN_DIM);
        for(int i=0; i<h2.rows; ++i) vDSP_vadd(h2.data.data() + i*HIDDEN_DIM, 1, b2.data.data(), 1, h2.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h2.data) v = std::tanh(v);
        
        Matrix y(x.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, OUTPUT_DIM, HIDDEN_DIM, 1.0f, h2.data.data(), HIDDEN_DIM, W_out.data.data(), OUTPUT_DIM, 0.0f, y.data.data(), OUTPUT_DIM);
        for(int i=0; i<y.rows; ++i) vDSP_vadd(y.data.data() + i*OUTPUT_DIM, 1, b_out.data.data(), 1, y.data.data() + i*OUTPUT_DIM, 1, OUTPUT_DIM);
        return y;
    }
};

std::vector<float> load_prices(const std::string& path) {
    std::vector<float> data;
    std::ifstream file(path);
    std::string line; std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); 
        std::getline(ss, cell, ','); 
        try { data.push_back(std::stof(cell)); } catch(...) {}
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string csv_path = argv[1];

    auto prices = load_prices(csv_path);
    if (prices.size() < INPUT_DIM + 10) return 1;

    std::vector<float> rets;
    for(size_t i=1; i<prices.size(); ++i) rets.push_back(std::log(prices[i]/prices[i-1]));

    int train_split = rets.size() * 0.8;
    FFNN64_64 model;
    auto start_time = std::chrono::steady_clock::now();
    int t = 1;
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 600) {
        int idx = rand() % (train_split - INPUT_DIM - 1);
        Matrix BX(BATCH_SIZE, INPUT_DIM), BY(BATCH_SIZE, OUTPUT_DIM);
        for(int b=0; b<BATCH_SIZE; ++b) {
            for(int j=0; j<INPUT_DIM; ++j) BX.data[b*INPUT_DIM+j] = rets[idx+b+j];
            BY.data[b] = rets[idx+b+INPUT_DIM];
        }
        Matrix pred = model.forward(BX);
        for(int b=0; b<BATCH_SIZE; ++b) {
            float d = pred.data[b] - BY.data[b];
            for(int i=0; i<INPUT_DIM; ++i) model.W1.grad[i] += d * 0.001f;
            for(int i=0; i<HIDDEN_DIM; ++i) {
                model.W2.grad[i] += d * 0.001f;
                model.W_out.grad[i] += d * 0.001f;
            }
        }
        radam_step(model.W1, 1e-3f, t);
        radam_step(model.W2, 1e-3f, t);
        radam_step(model.W_out, 1e-3f, t);
        t++;
    }

    float last_price = prices.back();
    std::vector<float> current_seq;
    for(int i=rets.size()-INPUT_DIM; i<(int)rets.size(); ++i) current_seq.push_back(rets[i]);

    std::cout << "DIA_FFNN_64_64 ";
    for(int year=1; year<=10; ++year) {
        float total_log_ret = 0;
        for(int d=0; d<252; ++d) {
            Matrix X(1, INPUT_DIM);
            X.data = current_seq;
            Matrix p = model.forward(X);
            float r = p.data[0];
            total_log_ret += r;
            current_seq.erase(current_seq.begin());
            current_seq.push_back(r);
        }
        last_price *= std::exp(total_log_ret);
        std::cout << last_price << (year == 10 ? "" : " ");
    }
    std::cout << std::endl;
    return 0;
}
