#include "embedding.hpp"
#include <cuda_runtime.h>
#include <cstdio> 

#define THREADS_PER_BLOCK 256

#define DBG(rank, ...) \
    do { \
        printf("[RANK %d] ", rank); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        fflush(stdout); \
    } while(0)

__global__ void embedding_lookup_kernel(
    const float* weight,
    const float* indices,
    float* output,
    int seq_len,
    int embedding_dim,
    int num_embeddings
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * embedding_dim;
    
    if (tid < total_elements) {
        int dim_idx = tid % embedding_dim;
        int seq_idx = tid / embedding_dim;
        
        int idx = static_cast<int>(indices[seq_idx]);
        
        if (idx >= 0 && idx < num_embeddings) {
            output[tid] = weight[idx * embedding_dim + dim_idx];
        } else {
            output[tid] = 0.0f;
        }
    }
}

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor({num_emb, emb_dim}, 0.0f);
}

Tensor Embedding::forward(const Tensor& indices) {
    int seq_len = indices.shape[0];
    //DBG(0, "seq_len=%d embedding_dim=%d", seq_len, embedding_dim);

    Tensor result({seq_len, embedding_dim});
    int total_output_elements = seq_len * embedding_dim;

    int weight_size = num_embeddings * embedding_dim;
    float *dev_weight, *dev_indices, *dev_output;

    cudaMalloc(&dev_weight, weight_size * sizeof(float));
    cudaMalloc(&dev_indices, seq_len * sizeof(float));
    cudaMalloc(&dev_output, total_output_elements * sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        //DBG(0, "ERROR: cudaMalloc failed: %s", cudaGetErrorString(err));
        return result;
    }

    cudaMemcpy(dev_weight, weight.data.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_indices, indices.data.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice);

    int nblks = (total_output_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    //DBG(0, "Launching kernel: %d blocks x %d threads", nblks, THREADS_PER_BLOCK);

    embedding_lookup_kernel<<<nblks, THREADS_PER_BLOCK>>>(dev_weight, dev_indices, dev_output, seq_len, embedding_dim, num_embeddings);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        //DBG(0, "ERROR: Kernel launch failed: %s", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        //DBG(0, "ERROR: Kernel execution failed: %s", cudaGetErrorString(err));
    }

    cudaMemcpy(result.data.data(), dev_output, total_output_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_weight);
    cudaFree(dev_indices);
    cudaFree(dev_output);

    //DBG(0, "Embedding forward complete");

    return result;
}