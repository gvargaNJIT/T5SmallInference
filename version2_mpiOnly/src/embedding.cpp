#include "embedding.hpp"
#include <mpi.h>
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

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor({num_emb, emb_dim}, 0.0f);
}

Tensor Embedding::forward(const Tensor& indices){
    MPI_Comm world = MPI_COMM_WORLD;
    int rank, num_procs;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &num_procs);

    int seq_len = indices.shape[0];
    // DBG(rank, "seq_len=%d num_procs=%d", seq_len, num_procs);

    int base  = seq_len / num_procs;
    int extra = seq_len % num_procs;

    int local_len = base + (rank < extra ? 1 : 0);
    int start     = rank * base + std::min(rank, extra);

    // DBG(rank, "start=%d local_len=%d", start, local_len);

    Tensor local_out({local_len, embedding_dim});
    for (int i = 0; i < local_len; i++) {
        int global_s = start + i;
        int idx = (int)indices.data[global_s];

        // DBG(rank, "lookup s=%d idx=%d", global_s, idx);

        const float* src = &weight.data[idx * embedding_dim];
        float* dst = &local_out.data[i * embedding_dim];

        std::copy(src, src + embedding_dim, dst);
    }

    // DBG(rank, "local loop done");

    std::vector<int> counts(num_procs), displs(num_procs);
    {
        int pos = 0;
        for (int r = 0; r < num_procs; r++) {
            int r_len = base + (r < extra ? 1 : 0);
            counts[r] = r_len * embedding_dim;
            displs[r] = pos * embedding_dim;

            // if (rank == ROOT)
            //     printf("[ROOT] gather r=%d r_len=%d displ=%d\n", r, r_len, displs[r]);

            pos += r_len;
        }
    }

    Tensor result({seq_len, embedding_dim});

    MPI_Gatherv(local_out.data.data(), local_len * embedding_dim, MPI_FLOAT, 
                result.data.data(), counts.data(), displs.data(), MPI_FLOAT, ROOT, world);

    MPI_Bcast(result.data.data(), result.size(), MPI_FLOAT, ROOT, world);

    return result;
}
