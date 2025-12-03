#include "rms_norm.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define THREADS_PER_BLOCK 256

#define DBG(rank, ...) \
    do { \
        printf("[RANK %d] ", rank); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        fflush(stdout); \
    } while(0)

__global__ void rmsnorm_kernel(
    const float* x,
    const float* weight,
    float* out,
    int seq_len,
    int hidden_size,
    float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[THREADS_PER_BLOCK];
    sdata[tid] = 0.0f;
    
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        float val = x[bid * hidden_size + h];
        sdata[tid] += val * val;
    }
    __syncthreads();
    __shared__ float shared_inv_rms;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < min(hidden_size, blockDim.x); i++) {
            sum_sq += sdata[i];
        }
        shared_inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    for (int h = tid; h < hidden_size; h += blockDim.x) {
        out[bid * hidden_size + h] = 
            x[bid * hidden_size + h] * shared_inv_rms * weight[h];
    }
}

RMSNorm::RMSNorm(int hidden_size, float epsilon)
    : eps(epsilon) {
    weight = Tensor({hidden_size}, 1.0f);
}

Tensor RMSNorm::forward(const Tensor& x) {
    int seq_len = x.shape[0];
    int hidden_size = x.shape[1];

    DBG(0, "RMSNorm: seq_len=%d hidden_size=%d eps=%f", seq_len, hidden_size, eps);
    
    Tensor result({seq_len, hidden_size});
    float *dev_input, *dev_weight, *dev_output;

    cudaMalloc(&dev_input, x.size() * sizeof(float));
    cudaMalloc(&dev_weight, weight.size() * sizeof(float));
    cudaMalloc(&dev_output, x.size() * sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        DBG(0, "ERROR: cudaMalloc failed: %s", cudaGetErrorString(err));
        return result;
    }

    cudaMemcpy(dev_input, x.data.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weight, weight.data.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);

    int nblks = seq_len;
    int nthds = THREADS_PER_BLOCK;

    DBG(0, "Launching kernel: %d blocks x %d threads", nblks, nthds);

    rmsnorm_kernel<<<nblks, nthds>>>(dev_input, dev_weight, dev_output, seq_len, hidden_size, eps);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        DBG(0, "ERROR: Kernel launch failed: %s", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        DBG(0, "ERROR: Kernel execution failed: %s", cudaGetErrorString(err));
    }

    cudaMemcpy(result.data.data(), dev_output, x.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_weight);
    cudaFree(dev_output);

    DBG(0, "RMSNorm forward complete");

    return result;
}