#include "embedding.hpp"
#include <cuda_runtime.h>
#include <mpi.h>
#include <vector>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#define ROOT 0

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
    int local_seq_len,
    int embedding_dim,
    int num_embeddings
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = local_seq_len * embedding_dim;

    if (tid < total) {
        int dim_idx = tid % embedding_dim;
        int seq_idx = tid / embedding_dim;
        int idx = (int)indices[seq_idx];

        if (idx >= 0 && idx < num_embeddings)
            output[tid] = weight[idx * embedding_dim + dim_idx];
        else
            output[tid] = 0.0f;
    }
}

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor({num_emb, emb_dim}, 0.0f);
}

Tensor Embedding::forward(const Tensor& indices) {
    MPI_Comm world = MPI_COMM_WORLD;
    int rank, num_procs;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &num_procs);

    int seq_len = indices.shape[0];
    // DBG(rank, "seq_len=%d num_procs=%d", seq_len, num_procs);

    // MPI_Bcast(weight.data.data(), num_embeddings * embedding_dim, MPI_FLOAT, ROOT, world);

    int base  = seq_len / num_procs;
    int extra = seq_len % num_procs;

    int local_len = base + (rank < extra ? 1 : 0);
    int start = rank * base + std::min(rank, extra);

    // DBG(rank, "start=%d local_len=%d", start, local_len);

    std::vector<float> local_idx(local_len);
    for (int i = 0; i < local_len; i++)
        local_idx[i] = indices.data[start + i];

    float *dev_w, *dev_idx, *dev_out;

    size_t w_size = num_embeddings * embedding_dim;
    size_t out_size = local_len * embedding_dim;

    cudaMalloc(&dev_w,   w_size * sizeof(float));
    cudaMalloc(&dev_idx, local_len * sizeof(float));
    cudaMalloc(&dev_out, out_size * sizeof(float));

    cudaMemcpy(dev_w, weight.data.data(), w_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_idx, local_idx.data(), local_len * sizeof(float), cudaMemcpyHostToDevice);

    int total = out_size;
    int nblks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    embedding_lookup_kernel<<<nblks, THREADS_PER_BLOCK>>>(dev_w, dev_idx, dev_out, local_len, embedding_dim, num_embeddings);

    Tensor local_out({local_len, embedding_dim});
    cudaMemcpy(local_out.data.data(), dev_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_w);
    cudaFree(dev_idx);
    cudaFree(dev_out);

    std::vector<int> counts(num_procs), displs(num_procs);
    {
        int pos = 0;
        for (int r = 0; r < num_procs; r++) {
            int r_len = base + (r < extra ? 1 : 0);
            counts[r] = r_len * embedding_dim;
            displs[r] = pos * embedding_dim;

            // if (rank == ROOT)
            //     printf("[ROOT] gather r=%d r_len=%d displ=%d\n",
            //            r, r_len, displs[r]);
            pos += r_len;
        }
    }

    Tensor result({seq_len, embedding_dim});

    MPI_Gatherv(local_out.data.data(), local_len * embedding_dim, MPI_FLOAT, 
                result.data.data(), counts.data(), displs.data(), MPI_FLOAT, ROOT, world);

    MPI_Bcast(result.data.data(), result.size(), MPI_FLOAT, ROOT, world);

    // DBG(rank, "After Gatherv");

    return result;
}
