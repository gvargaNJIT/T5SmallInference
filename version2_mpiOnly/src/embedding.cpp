#include "embedding.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor({num_emb, emb_dim});
    for (int i = 0; i < num_emb * emb_dim; i++) {
        weight.data[i] = 0.0f;
    }
}

Tensor Embedding::forward(const Tensor& indices) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int batch = indices.shape[0];
    int seq_len = indices.shape[1];
    int batch_per_proc = (batch + size - 1) / size;
    int start = std::min(rank * batch_per_proc, batch);
    int end = std::min(start + batch_per_proc, batch);
    int local_batch = end - start;
    std::vector<int> sendcounts(size), displs(size);
    for (int i = 0; i < size; i++) {
        int s = std::min(i * batch_per_proc, batch);
        int e = std::min(s + batch_per_proc, batch);
        
        sendcounts[i] = (e - s) * seq_len;
        displs[i] = s * seq_len;
    }

    std::vector<float> local_indices(local_batch * seq_len);
    MPI_Scatterv(rank == 0 ? indices.data.data() : nullptr,
                 sendcounts.data(),
                 displs.data(),
                 MPI_FLOAT,
                 local_indices.data(),
                 local_batch * seq_len,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    Tensor local_result({local_batch, seq_len, embedding_dim});
    for (int b = 0; b < local_batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int idx = static_cast<int>(std::round(local_indices[b * seq_len + s]));
            for (int d = 0; d < embedding_dim; d++) {
                local_result.data[b*seq_len*embedding_dim + s*embedding_dim + d] =
                    weight.data[idx*embedding_dim + d];
            }
        }
    }

    std::vector<int> recvcounts(size), displs_res(size);
    for (int i = 0; i < size; i++) {
        int s = std::min(i * batch_per_proc, batch);
        int e = std::min(s + batch_per_proc, batch);
        
        recvcounts[i] = (e - s) * seq_len * embedding_dim;
        displs_res[i] = s * seq_len * embedding_dim;
    }

    Tensor result;
    if (rank == 0) result = Tensor({batch, seq_len, embedding_dim});

    MPI_Gatherv(local_result.data.data(),
                local_result.data.size(),
                MPI_FLOAT,
                rank == 0 ? result.data.data() : nullptr,
                recvcounts.data(),
                displs_res.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    return result;
}