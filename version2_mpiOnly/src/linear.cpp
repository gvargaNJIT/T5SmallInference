#include "linear.hpp"
#include <cmath>
#include <algorithm>
#include <mpi.h>

Linear::Linear(int in_feat, int out_feat, bool use_bias_)
    : use_bias(use_bias_), in_features(in_feat), out_features(out_feat) {
    float std = std::sqrt(2.0f / (in_feat + out_feat));
    weight = Tensor::randn({in_features, out_features}, 0.0f, std); // **in x out** to match .T
    if (use_bias) {
        bias = Tensor::zeros({out_features});
    }
}

Tensor Linear::forward(const Tensor& x) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<int> batch_shape;
    for (size_t i = 0; i < x.shape.size() - 1; i++)
        batch_shape.push_back(x.shape[i]);

    int batch_size = 1;
    for (int dim : batch_shape) batch_size *= dim;

    if (x.shape.back() != in_features)
        throw std::runtime_error("Input last dimension mismatch with Linear in_features");

    Tensor x_2d = x.reshape({batch_size, in_features});
    Tensor local_out_2d({batch_size, out_features});
    int rows_per_rank = batch_size / size;
    int extra = batch_size % size;
    int start = rank * rows_per_rank + std::min(rank, extra);
    int count = rows_per_rank + (rank < extra ? 1 : 0);
    int end = start + count;
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
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; r++) {
        int rr = batch_size / size;
        int ee = batch_size % size;
        int c = rr + (r < ee ? 1 : 0);
        recv_counts[r] = c * out_features;
    }
    displs[0] = 0;
    for (int r = 1; r < size; r++)
        displs[r] = displs[r - 1] + recv_counts[r - 1];

    Tensor final_2d;
    if (rank == 0)
        final_2d = Tensor({batch_size, out_features});

    MPI_Gatherv(
        &local_out_2d.data[start * out_features],
        count * out_features,
        MPI_FLOAT,
        (rank == 0 ? final_2d.data.data() : nullptr),
        recv_counts.data(),
        displs.data(),
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        batch_shape.push_back(out_features);
        return final_2d.reshape(batch_shape);
    } else {
        return Tensor();
    }
}
