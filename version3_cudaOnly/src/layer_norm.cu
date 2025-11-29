#include "layer_norm.hpp"
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define THREADS_PER_BLOCK 1024

__global__ void layernorm_kernel(
    const float* x,
    const float* weight,
    float* out,
    int batch_size,
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

    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < min(hidden_size, blockDim.x); i++)
            sum_sq += sdata[i];
        float inv_rms = rsqrtf(sum_sq / hidden_size + eps);
        for (int h = 0; h < hidden_size; h++)
            out[bid * hidden_size + h] = x[bid * hidden_size + h] * inv_rms * weight[h];
    }
}

LayerNorm::LayerNorm(int hidden_size, float epsilon)
    : eps(epsilon) {
    weight = Tensor::ones({hidden_size});
}

Tensor LayerNorm::forward(const Tensor& x) {
    int hidden_size = x.shape[x.shape.size() - 1];
    int batch_size = x.size() / hidden_size;
    Tensor result(x.shape);
    float *dev_input, *dev_weight, *dev_output;

    cudaMalloc(&dev_input, x.size() * sizeof(float));
    cudaMalloc(&dev_weight, weight.size() * sizeof(float));
    cudaMalloc(&dev_output, x.size() * sizeof(float));

    cudaMemcpy(dev_input, x.data.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weight, weight.data.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);

    int nblks = batch_size;
    int nthds = THREADS_PER_BLOCK;
    size_t shared_mem = nthds * sizeof(float);

    layernorm_kernel<<<nblks, nthds, shared_mem>>>(dev_input,dev_weight,dev_output,batch_size,hidden_size,eps);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("RMSNorm kernel execution error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(result.data.data(), dev_output, x.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_weight);
    cudaFree(dev_output);

    return result;
}