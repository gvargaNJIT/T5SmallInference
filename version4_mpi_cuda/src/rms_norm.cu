#include "rms_norm.hpp"
#include <cuda_runtime.h>
#include <mpi.h>
#include <cmath>
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

__global__ void rmsnorm_kernel(
    const float* x,
    const float* weight,
    float* out,
    int local_seq_len,
    int hidden_size,
    float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float my_acc = 0.0f;

    for (int h = tid; h < hidden_size; h += blockDim.x) {
        float val = x[bid * hidden_size + h];
        my_acc += val * val;
    }

    sdata[tid] = my_acc;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }

    float inv_rms = 0.0f;
    if (tid == 0) {
        float sum_sq = sdata[0];
        inv_rms = rsqrtf(sum_sq / hidden_size + eps);
        sdata[0] = inv_rms;
    }
    __syncthreads();

    inv_rms = sdata[0];

    for (int h = tid; h < hidden_size; h += blockDim.x) {
        out[bid * hidden_size + h] =
            x[bid * hidden_size + h] * inv_rms * weight[h];
    }
}

RMSNorm::RMSNorm(int hidden_size, float epsilon)
    : eps(epsilon) {
    weight = Tensor({hidden_size}, 1.0f);
}

Tensor RMSNorm::forward(const Tensor& x) {
    MPI_Comm world = MPI_COMM_WORLD;
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int seq_len = x.shape[0];
    int hidden_size = x.shape[1];

    // DBG(rank, "seq_len=%d hidden_size=%d", seq_len, hidden_size);

    // Tensor x_bcast = x;
    // MPI_Bcast(x_bcast.data.data(), seq_len * hidden_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    // MPI_Bcast(weight.data.data(), hidden_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    int base  = seq_len / num_procs;
    int extra = seq_len % num_procs;
    int local_seq_len = base + (rank < extra ? 1 : 0);
    int start = rank * base + std::min(rank, extra);

    // DBG(rank, "start=%d local_seq_len=%d", start, local_seq_len);

    Tensor local_result({local_seq_len, hidden_size});

    if (local_seq_len > 0) {
        int total_elements = local_seq_len * hidden_size;
        float *dev_input = nullptr, *dev_weight = nullptr, *dev_output = nullptr;

        cudaError_t err;
        err = cudaMalloc(&dev_input, total_elements * sizeof(float));
        // if (err != cudaSuccess) {
            // DBG(rank, "cudaMalloc dev_input failed: %s", cudaGetErrorString(err));
        //     return local_result;
        // }

        err = cudaMalloc(&dev_weight, hidden_size * sizeof(float));
        // if (err != cudaSuccess) {
        //     DBG(rank, "cudaMalloc dev_weight failed: %s", cudaGetErrorString(err));
        //     cudaFree(dev_input);
        //     return local_result;
        // }

        err = cudaMalloc(&dev_output, total_elements * sizeof(float));
        // if (err != cudaSuccess) {
        //     DBG(rank, "cudaMalloc dev_output failed: %s", cudaGetErrorString(err));
        //     cudaFree(dev_input); cudaFree(dev_weight);
        //     return local_result;
        // }

        const float* src_ptr = x.data.data() + (size_t)start * hidden_size;
        err = cudaMemcpy(dev_input, src_ptr, total_elements * sizeof(float), cudaMemcpyHostToDevice);
        // if (err != cudaSuccess) {
        //     DBG(rank, "cudaMemcpy dev_input failed: %s", cudaGetErrorString(err));
        //     cudaFree(dev_input); cudaFree(dev_weight); cudaFree(dev_output);
        //     return local_result;
        // }

        err = cudaMemcpy(dev_weight, weight.data.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        // if (err != cudaSuccess) {
        //     DBG(rank, "cudaMemcpy dev_weight failed: %s", cudaGetErrorString(err));
        //     cudaFree(dev_input); cudaFree(dev_weight); cudaFree(dev_output);
        //     return local_result;
        // }

        int nblks = local_seq_len;
        // DBG(rank, "Launching kernel: %d blocks x %d threads", nblks, THREADS_PER_BLOCK);

        size_t shared_bytes = THREADS_PER_BLOCK * sizeof(float);
        rmsnorm_kernel<<<nblks, THREADS_PER_BLOCK, shared_bytes>>>(dev_input, dev_weight, dev_output, local_seq_len, hidden_size, eps);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            DBG(rank, "Kernel launch error: %s", cudaGetErrorString(err));
        }

        err = cudaDeviceSynchronize();
        // if (err != cudaSuccess) {
        //     DBG(rank, "Kernel execution error: %s", cudaGetErrorString(err));
        // }

        err = cudaMemcpy(local_result.data.data(), dev_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //     DBG(rank, "cudaMemcpy dev_output->host failed: %s", cudaGetErrorString(err));
        // }

        cudaFree(dev_input);
        cudaFree(dev_weight);
        cudaFree(dev_output);

        // DBG(rank, "Local CUDA done");
    } else {
        // DBG(rank, "No local rows assigned (local_seq_len==0)");
    }

    std::vector<int> counts(num_procs), displs(num_procs);
    {
        int pos = 0;
        for (int r = 0; r < num_procs; ++r) {
            int r_len = base + (r < extra ? 1 : 0);
            counts[r] = r_len * hidden_size;
            displs[r] = pos * hidden_size;
            // if (rank == ROOT) {
            //     printf("[ROOT] gather r=%d r_len=%d displ=%d\n", r, r_len, displs[r]);
            // }
            pos += r_len;
        }
    }

    // All ranks need to create the result tensor
    Tensor result({seq_len, hidden_size});
    
    MPI_Gatherv(local_result.data.data(), local_seq_len * hidden_size, MPI_FLOAT, 
                result.data.data(), counts.data(), displs.data(), MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    MPI_Bcast(result.data.data(), result.size(), MPI_FLOAT, ROOT, world);

    // DBG(rank, "After Gatherv - complete");

    return result;
}