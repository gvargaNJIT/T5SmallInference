#include "layer_norm.hpp"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

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

LayerNorm::LayerNorm(int dim, float epsilon){
    weight = Tensor({dim});
    for (int i = 0; i < dim; i++) {
        weight.data[i] = 1.0f;
    }
}

Tensor LayerNorm::forward(const Tensor& input) {
    MPI_Comm world = MPI_COMM_WORLD;
    int my_rank, num_procs;

    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int batch = input.shape[0];
    int seq_len = input.shape[1];
    int hidden_dim = input.shape[2];

    if (my_rank == ROOT) {
        DBG(my_rank, "input shape = (%d, %d, %d)", batch, seq_len, hidden_dim);
    }

    int my_work = (batch+num_procs-1)/num_procs;
    int start_idx = std::min(my_rank*my_work, batch);
    int end_idx = std::min(start_idx+my_work, batch);
    int n_local = end_idx - start_idx;
    std::vector<int> elms_to_comm(num_procs), offset_s(num_procs);
    int scatter_elms = n_local * seq_len * hidden_dim;
    DBG(my_rank, "start_idx=%d end_idx=%d n_local=%d scatter_elms=%d",start_idx, end_idx, n_local, scatter_elms);
    for (int i=0; i<num_procs; i++) {
        int proc_start = std::min(i*my_work, batch);
        int proc_end = std::min(proc_start+my_work, batch);
        elms_to_comm[i] = (proc_end-proc_start)*seq_len*hidden_dim;
        offset_s[i] = proc_start*seq_len*hidden_dim;
    }

    Tensor local_input({n_local, seq_len, hidden_dim});
    float *send_buf = nullptr;
    if (my_rank == ROOT) {
        send_buf = const_cast<float*>(input.data.data());
    }

    DBG(my_rank, "Scatterv: elms_to_comm[rank]=%d offset_s[rank]=%d (my part=%d)",elms_to_comm[my_rank], offset_s[my_rank], scatter_elms);

    MPI_Scatterv(send_buf,elms_to_comm.data(),offset_s.data(),MPI_FLOAT,local_input.data.data(),scatter_elms,MPI_FLOAT,ROOT,world);

    if (n_local > 0) {
        float v0 = local_input.data[0];
        DBG(my_rank, "local_input first element = %f", v0);
    }

    Tensor local_output({n_local, seq_len, hidden_dim});

    for (int b=0; b<n_local; b++) {
        for (int s=0; s<seq_len; s++) {
            int offset = b*seq_len*hidden_dim+s* hidden_dim;
            float sum_squares = 0.0f;
            for (int d = 0; d < hidden_dim; d++) {
                float val = local_input.data[offset + d];
                sum_squares += val * val;
            }

            float rms = std::sqrt(sum_squares / hidden_dim + eps);
            float inv_rms = 1.0f / rms;

            if (b == 0 && s == 0) {
                DBG(my_rank, "RMS first-token sum_squares=%f rms=%f", sum_squares, rms);
            }

            for (int d = 0; d < hidden_dim; d++) {
                local_output.data[offset + d] = 
                    local_input.data[offset + d] * inv_rms * weight.data[d];
            }
        }
    }

    Tensor result;
    if (my_rank == ROOT) result = Tensor({batch, seq_len, hidden_dim});

    float *recv_buf = nullptr;
    if (my_rank == ROOT) {
        recv_buf = result.data.data();
    }

    DBG(my_rank, "Gatherv: send_count=%zu recv_offset=%d",local_output.data.size(), offset_s[my_rank]);

    MPI_Gatherv(local_output.data.data(),local_output.data.size(),MPI_FLOAT,recv_buf,elms_to_comm.data(),offset_s.data(),MPI_FLOAT,ROOT,world);

    DBG(my_rank, "After Gatherv.");

    return result;
}
