// matmul.cpp
#include <mpi.h>
#include <vector>
#include <stdexcept>
#include "tensor.hpp"

#define ROOT 0

// Forward declare the CUDA function with correct signature
extern "C" Tensor matmul_cuda(const Tensor &a, const Tensor &b);

Tensor Tensor::matmul(const Tensor &other) const
{
    if (shape[1] != other.shape[0])
        throw std::runtime_error("matmul: shape mismatch");

    MPI_Comm world = MPI_COMM_WORLD;

    int my_rank, num_procs;
    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int m = shape[0];
    int n = shape[1];
    int p = other.shape[1];

    // Use consistent work distribution
    int base_work = m / num_procs;
    int extra = m % num_procs;
    
    int my_work = base_work + (my_rank < extra ? 1 : 0);
    int start_row = my_rank * base_work + std::min(my_rank, extra);

    // Extract this rank's rows
    Tensor my_rows({my_work, n});
    if (my_work > 0) {
        for (int i = 0; i < my_work; i++) {
            for (int j = 0; j < n; j++) {
                my_rows.data[i * n + j] = data[(start_row + i) * n + j];
            }
        }
    }

    // Use CUDA to compute this rank's portion
    Tensor C_local({my_work, p});
    
    if (my_work > 0) {
        C_local = matmul_cuda(my_rows, other);
    }

    // Gather results
    Tensor C({m, p});

    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);

    int pos = 0;
    for (int r = 0; r < num_procs; r++)
    {
        int rank_work = base_work + (r < extra ? 1 : 0);
        recvcounts[r] = rank_work * p;
        displs[r] = pos * p;
        pos += rank_work;
    }

    MPI_Gatherv(C_local.data.data(), my_work * p, MPI_FLOAT,
                C.data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, ROOT, world);

    MPI_Bcast(C.data.data(), m * p, MPI_FLOAT, ROOT, world);

    return C;
}