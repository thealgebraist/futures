#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <Accelerate/Accelerate.h>

const int INPUT_DIM = 20;
const int OUTPUT_DIM = 1;
const int NUM_SAMPLES = 10000;
const int BATCH_SIZE = 32;
const float CLIP_VAL = 5.0f;

struct Parameter {
    std::vector<float> v;    
    std::vector<float> g;    
    std::vector<float> grad_v;
    std::vector<float> grad_g;
    std::vector<float> m_v, v_v;
    std::vector<float> m_g, v_g;

    Parameter(int in, int out, float init_scale) {
        int size = in * out;
        v.resize(size);
        g.resize(out);
        grad_v.assign(size, 0.0f);
        grad_g.assign(out, 0.0f);
        m_v.assign(size, 0.0f);
        v_v.assign(size, 0.0f);
        m_g.assign(out, 0.0f);
        v_g.assign(out, 0.0f);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, init_scale);
        for (auto& val : v) val = dist(gen);
        
        for (int j = 0; j < out; ++j) {
            float norm = 0;
            vDSP_dotpr(v.data() + j * in, 1, v.data() + j * in, 1, &norm, in);
            g[j] = std::sqrt(norm) + 1e-8f;
        }
    }

    void compute_weights(std::vector<float>& w, int in, int out) {
        for (int j = 0; j < out; ++j) {
            float norm = 0;
            vDSP_dotpr(v.data() + j * in, 1, v.data() + j * in, 1, &norm, in);
            float scale = g[j] / (std::sqrt(norm) + 1e-8f);
            vDSP_vsmul(v.data() + j * in, 1, &scale, w.data() + j * in, 1, in);
        }
    }
};

void radam_step(std::vector<float>& p, std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v, float lr, int t) {
    float norm = 0;
    vDSP_dotpr(grad.data(), 1, grad.data(), 1, &norm, grad.size());
    norm = std::sqrt(norm);
    if (norm > CLIP_VAL) {
        float scale = CLIP_VAL / norm;
        vDSP_vsmul(grad.data(), 1, &scale, grad.data(), 1, grad.size());
    }

    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    for (size_t i = 0; i < p.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / (1.0f - std::pow(beta1, (float)t));
        float v_hat = std::sqrt(v[i] / (1.0f - std::pow(beta2, (float)t)));
        p[i] -= lr * m_hat / (v_hat + eps);
        grad[i] = 0;
    }
}

struct ScalingModel {
    int width;
    Parameter W1, W_out;
    std::vector<float> w1_flat, wout_flat;
    std::vector<float> h_batch, a_batch;

    ScalingModel(int n) : 
        width(n),
        W1(INPUT_DIM, n, 1.0f / std::sqrt((float)n)), 
        W_out(n, OUTPUT_DIM, 1.0f / (float)n),        
        w1_flat(INPUT_DIM * n),
        wout_flat(n * OUTPUT_DIM),
        h_batch(BATCH_SIZE * n), a_batch(BATCH_SIZE * n) {}

    void sync() {
        W1.compute_weights(w1_flat, INPUT_DIM, width);
        W_out.compute_weights(wout_flat, width, OUTPUT_DIM);
    }

    void forward_batch(const float* X_batch, float* Y_pred) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, BATCH_SIZE, width, INPUT_DIM, 1.0f, X_batch, INPUT_DIM, w1_flat.data(), INPUT_DIM, 0.0f, a_batch.data(), width);
        for(int i=0; i<BATCH_SIZE*width; ++i) h_batch[i] = std::tanh(a_batch[i]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, BATCH_SIZE, 1, width, 1.0f, h_batch.data(), width, wout_flat.data(), width, 0.0f, Y_pred, 1);
    }

    void backward_batch(const float* X_batch, const float* Y_pred, const float* Y_target) {
        for(int b=0; b<BATCH_SIZE; ++b) {
            float diff = (Y_pred[b] - Y_target[b]) / BATCH_SIZE;
            for(int j=0; j<width; ++j) {
                W_out.grad_v[j] += diff * h_batch[b*width + j];
                W_out.grad_g[0] += diff * h_batch[b*width + j];
                float grad_h = diff * wout_flat[j];
                float grad_a = grad_h * (1.0f - h_batch[b*width + j]*h_batch[b*width + j]);
                for(int i=0; i<INPUT_DIM; ++i) {
                    W1.grad_v[j*INPUT_DIM + i] += grad_a * X_batch[b*INPUT_DIM + i];
                    W1.grad_g[j] += grad_a * X_batch[b*INPUT_DIM + i];
                }
            }
        }
    }
};

float generate_target(int prob_idx, const float* x) {
    float y = 0;
    if (prob_idx % 4 == 0) {
        for(int i=0; i<INPUT_DIM-1; ++i) y += std::sin(x[i] * x[i+1]);
    } else if (prob_idx % 4 == 1) {
        for(int i=0; i<INPUT_DIM; ++i) y += std::cos(5.0f * x[i]);
    } else if (prob_idx % 4 == 2) {
        float den = 1.0f;
        for(int i=0; i<INPUT_DIM; ++i) den += std::abs(x[i]);
        y = 10.0f / den;
    } else {
        for(int i=0; i<INPUT_DIM; ++i) y += std::exp(-x[i]*x[i]);
    }
    return y;
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> X_all(NUM_SAMPLES * INPUT_DIM);
    std::vector<float> Y_all(NUM_SAMPLES);
    std::vector<float> XB(BATCH_SIZE * INPUT_DIM);
    std::vector<float> YB(BATCH_SIZE);
    std::vector<float> YP(BATCH_SIZE);

    for (int k = 0; k < 32; ++k) {
        for(int i=0; i<NUM_SAMPLES; ++i) {
            for(int j=0; j<INPUT_DIM; ++j) X_all[i*INPUT_DIM + j] = dist(gen);
            Y_all[i] = generate_target(k, &X_all[i*INPUT_DIM]);
        }

        int train_split = 8000;

        for (int n = 1; n <= 16; ++n) {
            ScalingModel model(n);
            auto start_time = std::chrono::steady_clock::now();
            int t = 1;
            int batches_done = 0;

            while (true) {
                if (batches_done % 100 == 0) {
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() >= 8) break;
                }
                model.sync();
                for(int b=0; b<BATCH_SIZE; ++b) {
                    int idx = rand() % train_split;
                    std::copy(X_all.begin() + idx*INPUT_DIM, X_all.begin() + (idx+1)*INPUT_DIM, XB.begin() + b*INPUT_DIM);
                    YB[b] = Y_all[idx];
                }
                model.forward_batch(XB.data(), YP.data());
                model.backward_batch(XB.data(), YP.data(), YB.data());
                radam_step(model.W1.v, model.W1.grad_v, model.W1.m_v, model.W1.v_v, 1e-3f, t);
                radam_step(model.W1.g, model.W1.grad_g, model.W1.m_g, model.W1.v_g, 1e-3f, t);
                radam_step(model.W_out.v, model.W_out.grad_v, model.W_out.m_v, model.W_out.v_v, 1e-3f, t);
                radam_step(model.W_out.g, model.W_out.grad_g, model.W_out.m_g, model.W_out.v_g, 1e-3f, t);
                t++;
                batches_done++;
            }

            model.sync();
            float mse = 0;
            std::vector<float> h_eval(n), a_eval(n);
            for (int i = train_split; i < NUM_SAMPLES; ++i) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, n, INPUT_DIM, 1.0f, model.w1_flat.data(), INPUT_DIM, &X_all[i*INPUT_DIM], 1, 0.0f, a_eval.data(), 1);
                for(int j=0; j<n; ++j) h_eval[j] = std::tanh(a_eval[j]);
                float p = 0;
                vDSP_dotpr(h_eval.data(), 1, model.wout_flat.data(), 1, &p, n);
                float d = p - Y_all[i];
                mse += d*d;
            }
            std::cout << k << " " << n << " " << (mse / (NUM_SAMPLES - train_split)) << std::endl;
        }
    }
    return 0;
}
