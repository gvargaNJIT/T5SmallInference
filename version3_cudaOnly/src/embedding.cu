#include "embedding.hpp"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define THREADS_PER_BLOCK 1024

__global__ void embedding_lookup_kernel(
    const float* weight,
    const float* indices,
    float* output,
    int batch,
    int seq_len,
    int embedding_dim,
    int num_embeddings
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * seq_len * embedding_dim;
    
    if (tid < total_elements) {
        int dim_idx = tid % embedding_dim;
        int seq_idx = (tid / embedding_dim) % seq_len;
        int batch_idx = tid / (seq_len * embedding_dim);
        int idx = static_cast<int>(indices[batch_idx * seq_len + seq_idx]);
        if (idx >= 0 && idx < num_embeddings) {
            output[tid] = weight[idx * embedding_dim + dim_idx];
        } else {
            output[tid] = 0.0f;
        }
    }
}

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor::randn({num_emb, emb_dim}, 0.0f, 0.02f);
}

Tensor Embedding::forward(const Tensor& indices) {
    int batch = indices.shape[0];
    int seq_len = indices.shape[1];
    Tensor result({batch, seq_len, embedding_dim});

    int total_output_elements = batch * seq_len * embedding_dim;
    int weight_size = num_embeddings * embedding_dim;
    int indices_size = batch * seq_len;
    float *dev_weight, *dev_indices, *dev_output;

    cudaMalloc(&dev_weight, weight_size * sizeof(float));
    cudaMalloc(&dev_indices, indices_size * sizeof(float));
    cudaMalloc(&dev_output, total_output_elements * sizeof(float));

    cudaMemcpy(dev_weight, weight.data.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_indices, indices.data.data(), indices_size * sizeof(float), cudaMemcpyHostToDevice);

    int nblks = (total_output_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int nthds = THREADS_PER_BLOCK;

    embedding_lookup_kernel<<<nblks, nthds>>>(dev_weight,dev_indices,dev_output,batch,seq_len,embedding_dim,num_embeddings);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(sync_err));
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    
    cudaMemcpy(result.data.data(), dev_output, total_output_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_weight);
    cudaFree(dev_indices);
    cudaFree(dev_output);

    return result;
}
