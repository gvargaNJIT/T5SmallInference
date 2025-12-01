#include "embedding.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>

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

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor({num_emb, emb_dim});
    for (int i = 0; i < num_emb * emb_dim; i++) {
        weight.data[i] = 0.0f;
    }
}

Tensor Embedding::forward(const Tensor& indices) {
    MPI_Comm world = MPI_COMM_WORLD;
    int my_rank, num_procs;
    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int batch = indices.shape[0];
    int seq_len = indices.shape[1];

    DBG(my_rank, "batch=%d seq_len=%d num_procs=%d", batch, seq_len, num_procs);

    int base = seq_len / num_procs;
    int extra = seq_len % num_procs;
    int local_seq_len = base + (my_rank < extra ? 1 : 0);
    int start_s = my_rank * base + std::min(my_rank, extra);

    DBG(my_rank, "SEQ_PAR: start_s=%d local_seq_len=%d", start_s, local_seq_len);

    std::vector<int> send_counts(num_procs), send_displs(num_procs);
    int total_send = 0;
    for (int r = 0; r < num_procs; ++r) {
        int r_len = base + (r < extra ? 1 : 0);
        send_counts[r] = batch * r_len;
        send_displs[r] = total_send;
        total_send += send_counts[r];
    }

    std::vector<int> local_idx(batch * local_seq_len, 0);

    std::vector<int> packed_sendbuf;
    int *sendbuf_ptr = nullptr;
    if (my_rank == ROOT) {
        packed_sendbuf.resize(total_send);
        for (int r = 0; r < num_procs; ++r) {
            int r_len = base + (r < extra ? 1 : 0);
            int r_start = r * base + std::min(r, extra);
            int out_off = send_displs[r];
            for (int b = 0; b < batch; ++b) {
                const float* src_float = &indices.data[b * seq_len + r_start];
                int dst_off = out_off + b * r_len;
                for (int t = 0; t < r_len; ++t) {
                    packed_sendbuf[dst_off + t] = static_cast<int>(std::lrint(src_float[t]));
                }
            }
        }
        sendbuf_ptr = packed_sendbuf.data();
    }

    MPI_Scatterv(sendbuf_ptr, send_counts.data(), send_displs.data(), MPI_INT, local_idx.data(), batch * local_seq_len, MPI_INT,ROOT, world);

    DBG(my_rank, "After Scatterv. Got %d ints.", (int)local_idx.size());

    Tensor local_out({batch, local_seq_len, embedding_dim});
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < local_seq_len; ++s) {
            int idx = local_idx[b * local_seq_len + s];
            if (idx < 0 || idx >= num_embeddings) {
                DBG(my_rank, "ERROR: bad idx %d at (b=%d,s=%d)", idx, b, s);
                idx = 0;
            }
            float* dst = &local_out.data[(b*local_seq_len + s) * embedding_dim];
            const float* src = &weight.data[idx * embedding_dim];
            std::copy(src, src + embedding_dim, dst);
        }
    }

    DBG(my_rank, "Local embedding lookup done.");

    std::vector<int> recv_counts(num_procs), recv_displs(num_procs);
    for (int r = 0; r < num_procs; ++r) {
        int r_len = base + (r < extra ? 1 : 0);
        recv_counts[r] = batch * r_len * embedding_dim;
        int r_start = r * base + std::min(r, extra);
        recv_displs[r] = batch * r_start * embedding_dim;
    }

    Tensor result;
    float* recvbuf = nullptr;
    if (my_rank == ROOT) {
        result = Tensor({batch, seq_len, embedding_dim});
        recvbuf = result.data.data();
    }

    MPI_Gatherv(local_out.data.data(), batch * local_seq_len * embedding_dim, MPI_FLOAT, recvbuf, recv_counts.data(), recv_displs.data(), MPI_FLOAT, ROOT, world);

    DBG(my_rank, "After Gatherv");

    return result;
}