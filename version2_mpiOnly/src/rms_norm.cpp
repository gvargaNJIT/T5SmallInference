#include "rms_norm.hpp"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <cstdio>

#define ROOT 0
#define DBG(rank, ...) \
    do { \
        printf("[RANK %d] ", rank); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        fflush(stdout); \
    } while(0)

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

    int base  = seq_len / num_procs;
    int extra = seq_len % num_procs;

    int local_seq = base + (rank < extra ? 1 : 0);
    int start     = rank * base + std::min(rank, extra);

    Tensor local_out({local_seq * hidden_size});
    const float* X = x.data.data();
    const float* W = weight.data.data();
    float* Y = local_out.data.data();

    for (int s = 0; s < local_seq; s++) {
        int global_s = start + s;
        const float* row = X + global_s * hidden_size;

        float variance = 0.0f;
        for (int h = 0; h < hidden_size; h++)
            variance += row[h] * row[h];

        variance /= hidden_size;
        float inv_std = 1.0f / std::sqrt(variance + eps);

        float* out_row = Y + s * hidden_size;
        for (int h = 0; h < hidden_size; h++)
            out_row[h] = row[h] * inv_std * W[h];
    }

    std::vector<int> counts(num_procs), displs(num_procs);

    int pos = 0;
    for (int r = 0; r < num_procs; r++) {
        int r_seq = base + (r < extra ? 1 : 0);
        counts[r] = r_seq * hidden_size;
        displs[r] = pos * hidden_size;
        pos += r_seq;
    }

    Tensor result({seq_len, hidden_size});

    MPI_Gatherv(local_out.data.data(), local_seq * hidden_size, MPI_FLOAT, 
                result.data.data(), counts.data(), displs.data(), MPI_FLOAT, ROOT, world);

    MPI_Bcast(result.data.data(), result.size(), MPI_FLOAT, ROOT, world);

    return result;
}
