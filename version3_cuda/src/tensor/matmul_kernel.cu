#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include "tensor.hpp"

#define MAX_TILE_WIDTH 32

__global__ void mat_mult_cuda(int m, int n, int p, const float *d_a, const float *d_b, float *d_c, int tile_width)
{
    __shared__ float a_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float b_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    float pValue = 0.0f;

    for (int phase = 0; phase < (n + tile_width - 1) / tile_width; ++phase)
    {
        int a_col = phase * tile_width + tx;
        int b_row = phase * tile_width + ty;

        if (row < m && a_col < n)
        {
            a_shared[ty][tx] = d_a[row * n + a_col];
        }
        else
        {
            a_shared[ty][tx] = 0.0f;
        }

        if (b_row < n && col < p)
        {
            b_shared[ty][tx] = d_b[b_row * p + col];
        }
        else
        {
            b_shared[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < tile_width; ++i)
        {
            pValue += a_shared[ty][i] * b_shared[i][tx];
        }

        __syncthreads();
    }
    
    if (row < m && col < p)
    {
        d_c[row * p + col] = pValue;
    }
}

extern "C" Tensor cuda_matmul(const Tensor& a, const Tensor& b)
{

    int m = a.shape[0];
    int n = a.shape[1];
    int p = b.shape[1];

    Tensor result({m, p});

    float *d_a, *d_b, *d_c;
    size_t size_a = m * n * sizeof(float);
    size_t size_b = n * p * sizeof(float);
    size_t size_c = m * p * sizeof(float);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, a.data.data(), size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data.data(), size_b, cudaMemcpyHostToDevice);

    int tile_width = 16;
    dim3 threadsPerBlock(tile_width, tile_width);
    dim3 numBlocks((p + tile_width - 1) / tile_width,
                   (m + tile_width - 1) / tile_width);

    mat_mult_cuda<<<numBlocks, threadsPerBlock>>>(m, n, p, d_a, d_b, d_c, tile_width);

    cudaMemcpy(result.data.data(), d_c, size_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
