#include "linear.hpp"
#include <cmath>
#include <algorithm>
#include <mpi.h>

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

Linear::Linear(int in_feat, int out_feat, bool use_bias_)
    : use_bias(use_bias_), in_features(in_feat), out_features(out_feat) {
    float std = std::sqrt(2.0f / (in_feat + out_feat));
    weight = Tensor::randn({in_features, out_features}, 0.0f, std); // **in x out** to match .T
    if (use_bias) {
        bias = Tensor::zeros({out_features});
    }
}

Tensor Linear::forward(const Tensor& x) {
    MPI_Comm world = MPI_COMM_WORLD;
    int my_rank, num_procs;

    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    std::vector<int> batch_shape;
    for (size_t i = 0; i < x.shape.size() - 1; i++)
        batch_shape.push_back(x.shape[i]);

    int batch_size = 1;
    for (int dim : batch_shape) batch_size *= dim;

    if (x.shape.back() != in_features)
        throw std::runtime_error("Input last dimension mismatch with Linear in_features");

    Tensor x_2d = x.reshape({batch_size, in_features});
    Tensor local_out_2d({batch_size, out_features});
    int my_work = batch_size / num_procs;
    int extra = batch_size % num_procs;
    int start = my_rank*my_work+std::min(my_rank, extra);
    int count = my_work + (my_rank < extra ? 1 : 0);
    int end = start + count;
    int count_out = count * out_features;

    DBG(my_rank, "batch_size=%d start=%d end=%d count=%d", batch_size, start, end, count);

    for (int b = start; b < end; b++) {
        for (int o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += x_2d.data[b * in_features + i] * weight.data[i * out_features + o];
            }
            if (use_bias) sum += bias.data[o];
            local_out_2d.data[b * out_features + o] = sum;
        }
    }

    if (count > 0)
        DBG(my_rank, "local_out_2d first element = %f", local_out_2d.data[start * out_features]);

    std::vector<int> recv_counts(num_procs);
    std::vector<int> offset(num_procs);
    for (int r = 0; r < num_procs; r++) {
        int rr = batch_size / num_procs;
        int ee = batch_size % num_procs;
        int c = rr + (r < ee ? 1 : 0);
        recv_counts[r] = c * out_features;
    }
    offset[0] = 0;
    for (int r = 1; r < num_procs; r++)
        offset[r] = offset[r - 1] + recv_counts[r - 1];

    Tensor final_2d;
    if (my_rank == ROOT)
        final_2d = Tensor({batch_size, out_features});

    float *recv_buf = nullptr;
    if (my_rank == ROOT) {
        recv_buf = final_2d.data.data();
        DBG(my_rank, "About to MPI_Gatherv: recv_counts[0]=%d offset[0]=%d", recv_counts[0], offset[0]);
    }

    float* local_ptr = &local_out_2d.data[start * out_features];

    MPI_Gatherv(local_ptr,count_out,MPI_FLOAT,recv_buf,recv_counts.data(),offset.data(),MPI_FLOAT,ROOT,world);

    if (my_rank == ROOT) {
        DBG(my_rank, "After MPI_Gatherv, final_2d shape = (%zu, %zu)", final_2d.shape[0], final_2d.shape[1]);
        batch_shape.push_back(out_features);
        return final_2d.reshape(batch_shape);
    } else {
        return Tensor();
    }
}
