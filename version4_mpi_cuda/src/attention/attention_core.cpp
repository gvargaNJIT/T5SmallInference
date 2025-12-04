#include <mpi.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "tensor.hpp"
#include "config.hpp"
#include "attention_mpi.hpp"

extern "C" Tensor matmul_cuda(const Tensor &a, const Tensor &b,int my_work);

static void compute_attention_head(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& bias,
    int h, int seq_len, int k_len, int d_kv,
    float* output_ptr)
{
    int q_head_size = seq_len * d_kv;
    int k_head_size = k_len * d_kv;
    int bias_head_size = seq_len * k_len;

    Tensor q_h({seq_len, d_kv});
    memcpy(q_h.data.data(),
        q.data.data() + h * q_head_size,
        q_head_size * sizeof(float));

    Tensor k_h({k_len, d_kv});
    memcpy(k_h.data.data(),
        k.data.data() + h * k_head_size,
        k_head_size * sizeof(float));

    Tensor v_h({k_len, d_kv});
    memcpy(v_h.data.data(),
        v.data.data() + h * k_head_size,
        k_head_size * sizeof(float));

    // Attention(Q,K,V) = Softmax((Q * K^T ) + B)*V
   
    Tensor scores = matmul_cuda(q_h, k_h.transpose(), seq_len);

    int bias_offset = h * bias_head_size;

    for (int i = 0; i < bias_head_size; ++i)
    {
        scores.data[i] += bias.data[bias_offset + i];
    }

    scores = scores.softmax();

    Tensor head_out = matmul_cuda(scores, v_h, k_len);

    memcpy(output_ptr, head_out.data.data(), q_head_size * sizeof(float));
}

Tensor compute_attention_parallel(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& bias,
    int n_heads, int seq_len, int k_len, int d_kv)
{
    MPI_Comm world = MPI_COMM_WORLD;

    int my_rank, num_procs;
    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int base_work = n_heads / num_procs;
    int my_work;
    int start_head;

    if (my_rank == num_procs - 1) {
        my_work = base_work + (n_heads % num_procs);
        start_head = base_work * my_rank;
    } else {
        my_work = base_work;
        start_head = base_work * my_rank;
    }

    int q_head_size = seq_len * d_kv;
    int my_total_size = my_work * q_head_size;
    Tensor local_context({my_total_size});

    for (int h = 0; h < my_work; h++) {
        int global_h = start_head + h;
        
        compute_attention_head(q, k, v, bias, global_h, seq_len, k_len, d_kv, 
                             local_context.data.data() + h * q_head_size);
    }

    Tensor context_layer({n_heads * seq_len * d_kv});

    std::vector<int> recvcounts;
    std::vector<int> displs;

    if (my_rank == 0) {
        recvcounts.resize(num_procs);
        displs.resize(num_procs);

        for (int r = 0; r < num_procs; r++) {
            int rank_work = (r == num_procs - 1) ? (base_work + (n_heads % num_procs)) : base_work;
            recvcounts[r] = rank_work * q_head_size;
            displs[r] = (base_work * r) * q_head_size;
        }
    }

    MPI_Gatherv(
        local_context.data.data(), my_total_size, MPI_FLOAT,
        context_layer.data.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
        0, world);

    MPI_Bcast(context_layer.data.data(), n_heads * seq_len * d_kv, MPI_FLOAT, 0, world);

    context_layer.shape = {n_heads, seq_len, d_kv};

    return context_layer;
}
