#include "rms_norm.hpp"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <mpi.h>

#include <cmath>
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

__global__ void rmsnorm_kernel(
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
        DBG_DEVICE("Starting rmsnorm: batch_size=%d hidden_size=%d", batch_size, hidden_size);
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

RMSNorm::RMSNorm(int hidden_size, float epsilon)
    : eps(epsilon) {
    weight = Tensor::ones({hidden_size});
}

Tensor RMSNorm::forward(const Tensor& input) {
    MPI_Comm world = MPI_COMM_WORLD;
    int my_rank, num_procs;
    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int batch = input.shape[0];
    int seq_len = input.shape[1];
    int hidden_dim = input.shape[2];

    if (my_rank == ROOT) {
        DBG(my_rank, "RMSNorm input shape = (%d, %d, %d) num_procs=%d", batch, seq_len, hidden_dim, num_procs);
    }

    int my_work = (batch + num_procs - 1) / num_procs;
    int start_idx = std::min(my_rank * my_work, batch);
    int end_idx = std::min(start_idx + my_work, batch);
    int n_local = end_idx - start_idx;

    std::vector<int> elms_to_comm(num_procs), offset_s(num_procs);
    int scatter_elms = n_local * seq_len * hidden_dim;
    
    DBG(my_rank, "start_idx=%d end_idx=%d n_local=%d scatter_elms=%d",start_idx, end_idx, n_local, scatter_elms);

    for (int i = 0; i < num_procs; i++) {
        int proc_start = std::min(i * my_work, batch);
        int proc_end = std::min(proc_start + my_work, batch);
        elms_to_comm[i] = (proc_end - proc_start) * seq_len * hidden_dim;
        offset_s[i] = proc_start * seq_len * hidden_dim;
    }

    Tensor local_input({n_local, seq_len, hidden_dim});
    float *send_buf = nullptr;
    if (my_rank == ROOT) {
        send_buf = const_cast<float*>(input.data.data());
    }

    DBG(my_rank, "Scatter: elms_to_comm[rank]=%d offset_s[rank]=%d",elms_to_comm[my_rank], offset_s[my_rank]);

    MPI_Scatterv(send_buf, elms_to_comm.data(), offset_s.data(), MPI_FLOAT,local_input.data.data(), scatter_elms, MPI_FLOAT, ROOT, world);

    if (n_local > 0) {
        DBG(my_rank, "After Scatter: local_input first element = %f", local_input.data[0]);
    }

    Tensor local_output({n_local, seq_len, hidden_dim});

    if (n_local > 0) {
        int cuda_batch_size = n_local * seq_len;
        int total_elements = cuda_batch_size * hidden_dim;
        float *dev_input, *dev_weight, *dev_output;

        DBG(my_rank, "Allocating CUDA memory: %d elements for input/output, %d for weights",total_elements, hidden_dim);

        cudaMalloc(&dev_input, total_elements * sizeof(float));
        cudaMalloc(&dev_weight, hidden_dim * sizeof(float));
        cudaMalloc(&dev_output, total_elements * sizeof(float));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: cudaMalloc failed: %s", cudaGetErrorString(err));
            return local_output;
        }

        cudaMemcpy(dev_input, local_input.data.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_weight, weight.data.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

        int nblks = cuda_batch_size;
        int nthds = THREADS_PER_BLOCK;

        DBG(my_rank, "Launching kernel: %d blocks x %d threads (cuda_batch_size=%d)",nblks, nthds, cuda_batch_size);

        rmsnorm_kernel<<<nblks, nthds>>>(dev_input, dev_weight, dev_output,cuda_batch_size, hidden_dim, eps);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: RMSNorm Kernel launch failed: %s", 
                cudaGetErrorString(err));
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            DBG(my_rank, "ERROR: RMSNorm Kernel execution failed: %s", cudaGetErrorString(err));
        }

        DBG(my_rank, "Kernel complete, copying back to host");

        cudaMemcpy(local_output.data.data(), dev_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dev_input);
        cudaFree(dev_weight);
        cudaFree(dev_output);

        DBG(my_rank, "Local CUDA RMSNorm complete");
    }

    Tensor result;
    if (my_rank == ROOT) {
        result = Tensor({batch, seq_len, hidden_dim});
    }

    float *recv_buf = nullptr;
    if (my_rank == ROOT) {
        recv_buf = result.data.data();
    }

    DBG(my_rank, "Gather: send_count=%zu recv_offset=%d",local_output.data.size(), offset_s[my_rank]);

    MPI_Gatherv(local_output.data.data(), local_output.data.size(), MPI_FLOAT,recv_buf, elms_to_comm.data(), offset_s.data(), MPI_FLOAT,ROOT, world);

    DBG(my_rank, "After Gather - RMSNorm forward complete");

    return result;
}