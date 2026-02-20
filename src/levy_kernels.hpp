#pragma once
#include <cmath>
#include <random>
#include <algorithm>

namespace levy {

const int INPUT_DIM = 16;
const int HIDDEN_DIM = 32;
const int OUTPUT_DIM = 1;
const int BATCH_SIZE = 32;
const float CLIP_VAL = 5.0f;
const float ALPHA = 1.5f;
const float SCALE = 0.005f;

inline float generate_stable(std::mt19937& gen) {
    std::uniform_real_distribution<float> u_dist(-M_PI/2.0f + 1e-5f, M_PI/2.0f - 1e-5f);
    std::exponential_distribution<float> w_dist(1.0f);
    float u = u_dist(gen);
    float w = w_dist(gen) + 1e-9f;
    float t1 = std::sin(ALPHA * u) / std::pow(std::cos(u), 1.0f/ALPHA);
    float t2 = std::pow(std::cos((1.0f - ALPHA) * u) / w, (1.0f - ALPHA) / ALPHA);
    return std::clamp(t1 * t2 * SCALE, -1.0f, 1.0f);
}

} // namespace levy
