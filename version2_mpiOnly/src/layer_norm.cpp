#include "layer_norm.hpp"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

LayerNorm::LayerNorm(int dim, float epsilon){
    weight = Tensor({dim});
    for (int i = 0; i < dim; i++) {
        weight.data[i] = 1.0f;
    }
}

Tensor LayerNorm::forward(const Tensor& input) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int batch = input.shape[0];
    int seq_len = input.shape[1];
    int hidden_dim = input.shape[2];
    int batch_per_proc = (batch + size - 1) / size;
    int start = std::min(rank * batch_per_proc, batch);
    int end = std::min(start + batch_per_proc, batch);
    int local_batch = end - start;
    std::vector<int> counts(size), displs(size);
    for (int i = 0; i < size; i++) {
        int s = std::min(i * batch_per_proc, batch);
        int e = std::min(s + batch_per_proc, batch);
        counts[i] = (e - s) * seq_len * hidden_dim;
        displs[i] = s * seq_len * hidden_dim;
    }

    Tensor local_input({local_batch, seq_len, hidden_dim});
    MPI_Scatterv(rank == 0 ? input.data.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_FLOAT,
                 local_input.data.data(),
                 local_batch * seq_len * hidden_dim,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    Tensor local_output({local_batch, seq_len, hidden_dim});

    for (int b = 0; b < local_batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int offset = b * seq_len * hidden_dim + s * hidden_dim;
            float sum_squares = 0.0f;
            for (int d = 0; d < hidden_dim; d++) {
                float val = local_input.data[offset + d];
                sum_squares += val * val;
            }

            float rms = std::sqrt(sum_squares / hidden_dim + eps);
            float inv_rms = 1.0f / rms;
            for (int d = 0; d < hidden_dim; d++) {
                local_output.data[offset + d] = 
                    local_input.data[offset + d] * inv_rms * weight.data[d];
            }
        }
    }

    Tensor result;
    if (rank == 0) result = Tensor({batch, seq_len, hidden_dim});
    MPI_Gatherv(local_output.data.data(),
                local_output.data.size(),
                MPI_FLOAT,
                rank == 0 ? result.data.data() : nullptr,
                counts.data(),
                displs.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    return result;
}
