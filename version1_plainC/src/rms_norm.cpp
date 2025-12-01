#include "rms_norm.hpp"
#include <cmath>
#include <cstdio>

RMSNorm::RMSNorm(int hidden_size, float epsilon)
    : eps(epsilon)
{
    weight = Tensor({hidden_size}, 0.0f);
}

Tensor RMSNorm::forward(const Tensor& x)
{
    int hidden_size = x.shape.back();

    Tensor result = x;

    int batch_size = x.size() / hidden_size;

    for (int b = 0; b < batch_size; b++) {

        float variance = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            int idx = b * hidden_size + h;
            variance += x.data[idx] * x.data[idx];
        }
        variance /= hidden_size;

        float inv_std = 1.0f / std::sqrt(variance + eps);

        for (int h = 0; h < hidden_size; h++) {
            int idx = b * hidden_size + h;
            result.data[idx] = x.data[idx] * inv_std * weight.data[h];
        }
    }

    return result;
}
