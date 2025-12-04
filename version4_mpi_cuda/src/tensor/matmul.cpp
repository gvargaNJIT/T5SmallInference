#include <mpi.h>
#include <vector>
#include <stdexcept>
#include "tensor.hpp"

#define ROOT 0

extern "C" Tensor matmul_cuda(const Tensor &a, const Tensor &b,int my_work);

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

    int base_work = m / num_procs;
    int my_work;
    int start_row;

    if (my_rank == num_procs - 1)
    {
        my_work = base_work + (m % num_procs);
        start_row = base_work * my_rank;
    }
    else
    {
        my_work = base_work;
        start_row = base_work * my_rank;
    }

    Tensor a_local({my_work, n});
    for (int i = 0; i < my_work; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a_local.data[i * n + j] = data[(start_row + i) * n + j];
        }
    }

    Tensor C_local = matmul_cuda(a_local, other, my_work);

    Tensor C({m, p});

    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);

    if (my_rank == ROOT)
    {
        for (int r = 0; r < num_procs; r++)
        {
            int rank_work = base_work;
            if (r == num_procs - 1)
                rank_work = base_work + (m % num_procs);
            recvcounts[r] = rank_work * p;
            displs[r] = (base_work * r) * p;
        }
    }

    MPI_Gatherv(C_local.data.data(), my_work * p, MPI_FLOAT,
                C.data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, ROOT, world);

    MPI_Bcast(C.data.data(), m * p, MPI_FLOAT, ROOT, world);

    return C;
}
