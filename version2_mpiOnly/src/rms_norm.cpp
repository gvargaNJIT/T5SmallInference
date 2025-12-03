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
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int hidden_size = x.shape.back();
    int batch_size  = x.size() / hidden_size;

    DBG(rank, "batch_size=%d hidden_size=%d", batch_size, hidden_size);
    Tensor x_bcast = x;
    MPI_Bcast(x_bcast.data.data(), batch_size * hidden_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    int base  = batch_size / num_procs;
    int extra = batch_size % num_procs;

    int local_batch = base + (rank < extra ? 1 : 0);
    int start       = rank * base + std::min(rank, extra);

    DBG(rank, "start=%d local_batch=%d", start, local_batch);

    Tensor local_out({local_batch * hidden_size});
    const float* X = x_bcast.data.data();
    const float* W = weight.data.data();
    float* Y = local_out.data.data();

    for (int b = 0; b < local_batch; b++) {
        int global_b = start + b;
        const float* row = X + global_b * hidden_size;

        // variance = mean(x²)
        float variance = 0.0f;
        for (int h = 0; h < hidden_size; h++)
            variance += row[h] * row[h];

        variance /= hidden_size;
        float inv_std = 1.0f / std::sqrt(variance + eps);

        if (b == 0)
            DBG(rank, "First row: variance=%f inv_std=%f", variance, inv_std);

        // normalize + scale
        float* out_row = Y + b * hidden_size;
        for (int h = 0; h < hidden_size; h++)
            out_row[h] = row[h] * inv_std * W[h];
    }

    DBG(rank, "Compute done");

    std::vector<int> counts(num_procs), displs(num_procs);

    {
        int pos = 0;
        for (int r = 0; r < num_procs; r++) {
            int r_batch = base + (r < extra ? 1 : 0);

            counts[r] = r_batch * hidden_size;
            displs[r] = pos * hidden_size;

            if (rank == ROOT)
                printf("[ROOT] gather r=%d batch=%d displ=%d\n",
                        r, r_batch, displs[r]);

            pos += r_batch;
        }
    }

    Tensor result;
    float* recvbuf = nullptr;

    if (rank == ROOT) {
        result = Tensor({batch_size * hidden_size});
        recvbuf = result.data.data();
    }

    MPI_Gatherv(local_out.data.data(), local_batch * hidden_size, MPI_FLOAT, recvbuf, counts.data(), displs.data(), MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    DBG(rank, "After Gatherv");

    return result;
}
