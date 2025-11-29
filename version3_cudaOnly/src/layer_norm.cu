#include "layer_norm.hpp"
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define THREADS_PER_BLOCK 1024

#define DBG(rank, ...) \
    do { \
        printf("[RANK %d] ", rank); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        fflush(stdout); \
    } while(0)

#define DBG_DEVICE(...) \
    do { \
        printf("[DEVICE blockIdx=(%d,%d) threadIdx=(%d,%d)] ", \
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } while(0)

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

    if (bid == 0 && tid == 0) {
        DBG_DEVICE("Starting layernorm: batch_size=%d hidden_size=%d", batch_size, hidden_size);
    }

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
        for (int i = 0; i < min(hidden_size, blockDim.x); i++)
            sum_sq += sdata[i];
        shared_inv_rms = rsqrtf(sum_sq / hidden_size + eps);
        if (bid == 0) {
            DBG_DEVICE("Block %d: sum_sq=%f inv_rms=%f", bid, sum_sq, shared_inv_rms);
        }
    }
    __syncthreads();
    
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        out[bid * hidden_size + h] = x[bid * hidden_size + h] * shared_inv_rms * weight[h];
        if (bid == 0 && h < 3) {
            DBG_DEVICE("out[%d][%d] = %f", bid, h, out[bid * hidden_size + h]);
        }
    }
}

LayerNorm::LayerNorm(int hidden_size, float epsilon)
    : eps(epsilon) {
    weight = Tensor::ones({hidden_size});
}

Tensor LayerNorm::forward(const Tensor& x) {
    int hidden_size = x.shape[x.shape.size() - 1];
    int batch_size = x.size() / hidden_size;

    DBG(0, "LayerNorm: batch_size=%d hidden_size=%d eps=%f", batch_size, hidden_size, eps);
    
    Tensor result(x.shape);
    float *dev_input, *dev_weight, *dev_output;

    DBG(0, "Allocating %zu floats for input/output, %zu for weights", x.size(), weight.size());

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

    int nblks = batch_size;
    int nthds = THREADS_PER_BLOCK;

    DBG(0, "Launching kernel: %d blocks x %d threads", nblks, nthds);

    layernorm_kernel<<<nblks, nthds>>>(dev_input,dev_weight,dev_output,batch_size,hidden_size,eps);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        DBG(0, "ERROR: RMSNorm Kernel launch failed: %s", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        DBG(0, "ERROR: RMSNorm Kernel execution failed: %s", cudaGetErrorString(err));
    }

    DBG(0, "Kernel complete, copying back to host");

    cudaMemcpy(result.data.data(), dev_output, x.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_weight);
    cudaFree(dev_output);

    DBG(0, "LayerNorm forward complete");

    return result;
}