#include "rms_norm.hpp"
#include <cmath>

RMSNorm::RMSNorm(int hidden_size, float epsilon)
    : eps(epsilon)
{
    weight = Tensor({hidden_size}, 1.0f);
}

Tensor RMSNorm::forward(const Tensor& x)
{
    int seq_len = x.shape[0];
    int hidden_size = x.shape[1];
    
    Tensor result({seq_len, hidden_size});
    
    for (int s = 0; s < seq_len; s++) {
        float variance = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            int idx = s * hidden_size + h;
            variance += x.data[idx] * x.data[idx];
        }
        variance /= hidden_size;
        float inv_std = 1.0f / std::sqrt(variance + eps);
        
        for (int h = 0; h < hidden_size; h++) {
            int idx = s * hidden_size + h;
            result.data[idx] = x.data[idx] * inv_std * weight.data[h];
        }
    }
    
    return result;
}