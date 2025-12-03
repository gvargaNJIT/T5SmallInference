#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>
#include "tensor.hpp"

#define BLOCK_SIZE 256

__global__ void softmax_kernel(const float* input, float* output,
                               int rows_count, int row_length)
{
    int row = blockIdx.x;
    
    if (row >= rows_count)
        return;

    int offset = row * row_length;

    __shared__ float shared_max;
    __shared__ float shared_sum;

    float local_max = 0;

    for (int i = threadIdx.x; i < row_length; i += blockDim.x)
    {
        local_max = fmaxf(local_max, input[offset + i]);
    }

    __shared__ float max_vals[BLOCK_SIZE];
    max_vals[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            max_vals[threadIdx.x] = fmaxf(max_vals[threadIdx.x], max_vals[threadIdx.x + stride]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        shared_max = max_vals[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_length; i += blockDim.x)
    {
        float exp_val = expf(input[offset + i] - shared_max);
        output[offset + i] = exp_val;
        local_sum += exp_val;
    }

    __shared__ float sum_vals[BLOCK_SIZE];
    sum_vals[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sum_vals[threadIdx.x] += sum_vals[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        shared_sum = sum_vals[0];
    __syncthreads();

    for (int i = threadIdx.x; i < row_length; i += blockDim.x)
    {
        output[offset + i] /= shared_sum;
    }
}

extern "C" Tensor cuda_softmax(const Tensor& input)
{
    int row_length = input.shape[input.shape.size() - 1];
    int rows_count = input.size() / row_length;

    Tensor output(input.shape);

    float *d_input, *d_output;
    size_t size = input.size() * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = rows_count;

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, rows_count, row_length);

    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
