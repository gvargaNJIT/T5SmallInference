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

    DBG(my_rank, "batch=%d seq_len=%d embedding_dim=%d num_procs=%d", 
        batch, seq_len, embedding_dim, num_procs);

    int my_work = (batch + num_procs - 1) / num_procs;
    int start_idx = std::min(my_rank * my_work, batch);
    int end_idx = std::min(start_idx + my_work, batch);
    int n_local = end_idx - start_idx;

    DBG(my_rank, "my_work=%d start_idx=%d end_idx=%d n_local=%d",
        my_work, start_idx, end_idx, n_local);

    std::vector<int> elms_to_comm(num_procs), offset(num_procs);
    int scatter_elms = n_local * seq_len;
    for (int i = 0; i < num_procs; i++) {
        int proc_start = std::min(i * my_work, batch);
        int proc_end = std::min(proc_start + my_work, batch);
        elms_to_comm[i] = (proc_end - proc_start) * seq_len;
        offset[i] = proc_start * seq_len;
    }

    if (my_rank == ROOT) {
        for (int i = 0; i < num_procs; i++) {
            DBG(ROOT, "Scatter: rank %d gets %d elements at offset %d",
                i, elms_to_comm[i], offset[i]);
        }
    }

    std::vector<float> local_indices(n_local * seq_len);
    float *send_buf = nullptr;
    if (my_rank == ROOT) {
        send_buf = const_cast<float*>(indices.data.data());
    }

    MPI_Scatterv(send_buf, elms_to_comm.data(), offset.data(), MPI_FLOAT,local_indices.data(), scatter_elms, MPI_FLOAT, ROOT, world);

    DBG(my_rank, "After Scatter: received %d floats", scatter_elms);

    Tensor local_result({n_local, seq_len, embedding_dim});
    
    if (n_local > 0) {
        int total_output_elements = n_local * seq_len * embedding_dim;
        int weight_size = num_embeddings * embedding_dim;
        int indices_size = n_local * seq_len;

        float *dev_weight, *dev_indices, *dev_output;

        cudaMalloc(&dev_weight, weight_size * sizeof(float));
        cudaMalloc(&dev_indices, indices_size * sizeof(float));
        cudaMalloc(&dev_output, total_output_elements * sizeof(float));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMalloc failed: %s", cudaGetErrorString(err));
            return local_result;
        }

        cudaMemcpy(dev_weight, weight.data.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_indices, local_indices.data(), indices_size * sizeof(float), cudaMemcpyHostToDevice);

        int nblks = (total_output_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int nthds = THREADS_PER_BLOCK;
        DBG(my_rank, "Launching kernel: %d blocks x %d threads", nblks, nthds);

        embedding_lookup_kernel<<<nblks, nthds>>>(
            dev_weight, dev_indices, dev_output,
            n_local, seq_len, embedding_dim, num_embeddings
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: Kernel launch failed: %s", cudaGetErrorString(err));
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: Kernel execution failed: %s", cudaGetErrorString(err));
        }

        DBG(my_rank, "Kernel complete, copying back to host");

        cudaMemcpy(local_result.data.data(), dev_output, total_output_elements * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dev_weight);
        cudaFree(dev_indices);
        cudaFree(dev_output);

        DBG(my_rank, "Local CUDA embedding lookup done.");
    }

    std::vector<int> elms_to_gather(num_procs), offset_res(num_procs);
    for (int i = 0; i < num_procs; i++) {
        int s = std::min(i * my_work, batch);
        int e = std::min(s + my_work, batch);
        elms_to_gather[i] = (e - s) * seq_len * embedding_dim;
        offset_res[i] = s * seq_len * embedding_dim;
    }

    Tensor result;
    if (my_rank == ROOT) {
        result = Tensor({batch, seq_len, embedding_dim});
    }

    float *recv_buf = nullptr;
    if (my_rank == ROOT) {
        recv_buf = result.data.data();
    }

    MPI_Gatherv(local_result.data.data(), local_result.data.size(), MPI_FLOAT,recv_buf, elms_to_gather.data(), offset_res.data(), MPI_FLOAT,ROOT, world);

    DBG(my_rank, "After Gather - Embedding forward complete");

    return result;
}