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

struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}
    static Matrix random(int r, int c, float scale=0.01f) {
        Matrix mat(r, c);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& val : mat.data) val = dist(gen);
        return mat;
    }
};

struct FFNN64 {
    Matrix W1, b1, W_out, b_out;
    FFNN64() : 
        W1(Matrix::random(INPUT_DIM, HIDDEN_DIM)), 
        b1(Matrix::random(1, HIDDEN_DIM)),
        W_out(Matrix::random(HIDDEN_DIM, OUTPUT_DIM)),
        b_out(Matrix::random(1, OUTPUT_DIM)) {}

    Matrix forward(const Matrix& x) {
        Matrix h(x.rows, HIDDEN_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, HIDDEN_DIM, INPUT_DIM, 1.0f, x.data.data(), INPUT_DIM, W1.data.data(), HIDDEN_DIM, 0.0f, h.data.data(), HIDDEN_DIM);
        for(int i=0; i<h.rows; ++i) vDSP_vadd(h.data.data() + i*HIDDEN_DIM, 1, b1.data.data(), 1, h.data.data() + i*HIDDEN_DIM, 1, HIDDEN_DIM);
        for(auto& v : h.data) v = std::tanh(v);
        Matrix y(x.rows, OUTPUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.rows, OUTPUT_DIM, HIDDEN_DIM, 1.0f, h.data.data(), HIDDEN_DIM, W_out.data.data(), OUTPUT_DIM, 0.0f, y.data.data(), OUTPUT_DIM);
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

float calc_mse(FFNN64& model, const std::vector<float>& rets, int n) {
    float mse = 0;
    int count = 0;
    for(int i=0; i<n-INPUT_DIM-1; i += 100) { // Subsample for speed
        Matrix X(1, INPUT_DIM);
        for(int j=0; j<INPUT_DIM; ++j) X.data[j] = rets[i+j];
        Matrix p = model.forward(X);
        float d = p.data[0] - rets[i+INPUT_DIM];
        mse += d*d;
        count++;
    }
    return mse / count;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::string csv_path = argv[1];
    std::string symbol = argv[2];

    auto prices = load_prices(csv_path);
    if (prices.size() < INPUT_DIM + 10) return 1;
    std::vector<float> rets;
    for(size_t i=1; i<prices.size(); ++i) rets.push_back(std::log(prices[i]/prices[i-1]));

    int train_split = rets.size() * 0.8;
    FFNN64 best_model;
    float best_mse = calc_mse(best_model, rets, train_split);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.001f);

    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() < 300) {
        FFNN64 candidate = best_model;
        // Perturb
        for(auto& v : candidate.W1.data) v += dist(gen);
        for(auto& v : candidate.W_out.data) v += dist(gen);
        
        float mse = calc_mse(candidate, rets, train_split);
        if(mse < best_mse) {
            best_mse = mse;
            best_model = candidate;
        }
    }

    float last_price = prices.back();
    std::vector<float> current_seq;
    for(int i=rets.size()-INPUT_DIM; i<(int)rets.size(); ++i) current_seq.push_back(rets[i]);

    std::cout << symbol << " ";
    for(int year=1; year<=10; ++year) {
        float total_log_ret = 0;
        for(int d=0; d<252; ++d) {
            Matrix X(1, INPUT_DIM);
            X.data = current_seq;
            Matrix p = best_model.forward(X);
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
