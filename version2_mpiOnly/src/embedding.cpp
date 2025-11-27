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

    int my_work = (batch+num_procs-1)/num_procs;
    int start_idx = std::min(my_rank*my_work, batch);
    int end_idx = std::min(start_idx+my_work, batch);
    int n_local = end_idx - start_idx;

    DBG(my_rank, "my_work=%d start_idx=%d end_idx=%d n_local=%d",my_work, start_idx, end_idx, n_local);

    std::vector<int> elms_to_comm(num_procs), offset(num_procs);
    int scatter_elms = n_local * seq_len;
    for (int i=0; i<num_procs; i++) {
        int proc_start = std::min(i*my_work, batch);
        int proc_end = std::min(proc_start+my_work, batch);
        
        elms_to_comm[i] = (proc_end-proc_start) * seq_len;
        offset[i] = proc_start * seq_len;
    }

    if (my_rank == ROOT) {
        for (int i=0; i<num_procs; i++) {
            DBG(ROOT, "Scatter: rank %d gets %d elements at offset %d",
                i, elms_to_comm[i], offset[i]);
        }
    }

    std::vector<float> local_indices(n_local*seq_len);
    float *send_buf = nullptr;
    if (my_rank == ROOT) {
        send_buf = const_cast<float*>(indices.data.data());
    }

    MPI_Scatterv(send_buf,elms_to_comm.data(),offset.data(),MPI_FLOAT,local_indices.data(),scatter_elms,MPI_FLOAT,ROOT,world);

    DBG(my_rank, "After Scatterv: received %d floats", scatter_elms);

    Tensor local_result({n_local, seq_len, embedding_dim});
    for (int b=0; b<n_local; b++) {
        for (int s=0; s<seq_len; s++) {
            int idx=static_cast<int>(std::round(local_indices[b*seq_len+s]));
            if (idx < 0 || idx >= num_embeddings) {
                DBG(my_rank, "ERROR: invalid index %d at local pos (b=%d,s=%d)", idx, b, s);
            }
            for (int d=0; d<embedding_dim; d++) {
                local_result.data[b*seq_len*embedding_dim + s*embedding_dim + d] =
                    weight.data[idx*embedding_dim + d];
            }
        }
    }
    DBG(my_rank, "Local embedding lookup done.");

    std::vector<int> elms_to_gather(num_procs), offset_res(num_procs);
    for (int i=0; i<num_procs; i++) {
        int s = std::min(i*my_work, batch);
        int e = std::min(s+my_work, batch);
        
        elms_to_gather[i] = (e-s)*seq_len*embedding_dim;
        offset_res[i] = s*seq_len*embedding_dim;
    }

    Tensor result;
    if (my_rank == ROOT) result = Tensor({batch, seq_len, embedding_dim});

    float *recv_buf = nullptr;
    if (my_rank == ROOT) {
        recv_buf = result.data.data();
    }

    MPI_Gatherv(local_result.data.data(),local_result.data.size(),MPI_FLOAT,recv_buf,elms_to_gather.data(),offset_res.data(),MPI_FLOAT,ROOT,world);

    DBG(my_rank, "After Gatherv.");

    return result;
}