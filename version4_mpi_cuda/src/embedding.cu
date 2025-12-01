#include "embedding.hpp"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <mpi.h>

#include <vector>
#include <algorithm>

#define THREADS_PER_BLOCK 1024
#define COLOR 1<<10
#define MAXDIM 1<<12
#define ROOT 0

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

__global__ void embedding_lookup_kernel(
    const float* weight,
    const float* indices,
    float* output,
    int batch,
    int local_seq_len,
    int embedding_dim,
    int num_embeddings
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * local_seq_len * embedding_dim;
    
    if (tid < total_elements) {
        int dim_idx   = tid % embedding_dim;
        int seq_idx   = (tid / embedding_dim) % local_seq_len;
        int batch_idx = tid / (local_seq_len * embedding_dim);
        int idx = static_cast<int>(indices[batch_idx * local_seq_len + seq_idx]);
        if (idx >= 0 && idx < num_embeddings) {
            output[tid] = weight[idx * embedding_dim + dim_idx];
        } else {
            output[tid] = 0.0f;
            if (tid < 3) {
                DBG_DEVICE("ERROR: invalid index %d at tid=%d", idx, tid);
            }
        }
    }
}

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor::randn({num_emb, emb_dim}, 0.0f, 0.02f);
}

Tensor Embedding::forward(const Tensor& indices) {
    MPI_Comm world = MPI_COMM_WORLD;
    int my_rank, num_procs;
    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int batch = indices.shape[0];
    int seq_len = indices.shape[1];

    DBG(my_rank, "batch=%d seq_len=%d embedding_dim=%d num_procs=%d", batch, seq_len, embedding_dim, num_procs);

    int base = seq_len / num_procs;
    int extra = seq_len % num_procs;
    int local_seq_len = base + (my_rank < extra ? 1 : 0);
    int start_s = my_rank * base + std::min(my_rank, extra);
    DBG(my_rank, "SEQ_SPLIT: start_s=%d local_seq_len=%d", start_s, local_seq_len);

    int n_local = (local_seq_len > 0) ? batch : 0;

    std::vector<int> send_counts(num_procs), send_displs(num_procs);
    for (int r = 0; r < num_procs; ++r) {
        int r_len = base + (r < extra ? 1 : 0);
        send_counts[r] = batch * r_len;
        int r_start = r * base + std::min(r, extra);
        send_displs[r] = batch * r_start;
    }

    std::vector<int> local_indices(batch * local_seq_len);
    float *send_buf = nullptr;
    if (my_rank == ROOT) {
        send_buf = const_cast<float*>(indices.data.data());
    }

    MPI_Scatterv(send_buf, send_counts.data(), send_displs.data(), MPI_FLOAT, local_indices.data(), batch * local_seq_len, MPI_FLOAT, ROOT, world);

    DBG(my_rank, "After Scatterv: received %d floats (batch*local_seq_len=%d)", (int)local_indices.size(), batch * local_seq_len);

    Tensor local_result({n_local, local_seq_len, embedding_dim});

    if (n_local > 0 && local_seq_len > 0) {
        int total_output_elements = n_local * local_seq_len * embedding_dim;
        int weight_size = num_embeddings * embedding_dim;
        int indices_size = batch * local_seq_len;

        float *dev_weight = nullptr, *dev_indices = nullptr, *dev_output = nullptr;

        cudaError_t err;
        if ((err = cudaMalloc(&dev_weight, weight_size * sizeof(float))) != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMalloc(dev_weight) failed: %s", cudaGetErrorString(err));
            return local_result;
        }
        if ((err = cudaMalloc(&dev_indices, indices_size * sizeof(float))) != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMalloc(dev_indices) failed: %s", cudaGetErrorString(err));
            cudaFree(dev_weight);
            return local_result;
        }
        if ((err = cudaMalloc(&dev_output, total_output_elements * sizeof(float))) != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMalloc(dev_output) failed: %s", cudaGetErrorString(err));
            cudaFree(dev_weight); cudaFree(dev_indices);
            return local_result;
        }

        err = cudaMemcpy(dev_weight, weight.data.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMemcpy(weight) failed: %s", cudaGetErrorString(err));
        }
        err = cudaMemcpy(dev_indices, local_indices.data(), indices_size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMemcpy(indices) failed: %s", cudaGetErrorString(err));
        }

        int nblks = (total_output_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int nthds = THREADS_PER_BLOCK;
        DBG(my_rank, "Launching kernel: %d blocks x %d threads (total_output=%d)", nblks, nthds, total_output_elements);

        embedding_lookup_kernel<<<nblks, nthds>>>(dev_weight, dev_indices, dev_output, n_local, local_seq_len, embedding_dim, num_embeddings);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: Kernel launch failed: %s", cudaGetErrorString(err));
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: Kernel execution failed: %s", cudaGetErrorString(err));
        }

        err = cudaMemcpy(local_result.data.data(), dev_output, total_output_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMemcpy(dev_output->host) failed: %s", cudaGetErrorString(err));
        }

        cudaFree(dev_weight);
        cudaFree(dev_indices);
        cudaFree(dev_output);

        DBG(my_rank, "Local CUDA embedding lookup done (batch=%d local_seq_len=%d)", n_local, local_seq_len);
    } 
    else {
        DBG(my_rank, "No local sequence assigned (local_seq_len=%d)", local_seq_len);
    }

    std::vector<int> recv_counts(num_procs), recv_displs(num_procs);
    for (int r = 0; r < num_procs; ++r) {
        int r_len = base + (r < extra ? 1 : 0);
        recv_counts[r] = batch * r_len * embedding_dim;
        int r_start = r * base + std::min(r, extra);
        recv_displs[r] = batch * r_start * embedding_dim;
    }

    Tensor result;
    float *recv_buf = nullptr;
    if (my_rank == ROOT) {
        result = Tensor({batch, seq_len, embedding_dim});
        recv_buf = result.data.data();
    }

    MPI_Gatherv(local_result.data.data(), n_local * local_seq_len * embedding_dim,MPI_FLOAT, recv_buf, recv_counts.data(), recv_displs.data(), MPI_FLOAT,ROOT, world);

    DBG(my_rank, "After Gatherv - Embedding forward complete");

    return result;
}
