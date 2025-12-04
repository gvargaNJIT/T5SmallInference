#include <mpi.h>
#include <cmath>
#include <vector>
#include "tensor.hpp"


#define ROOT 0


Tensor softmax(Tensor a) 
{
    MPI_Comm world = MPI_COMM_WORLD;

    int my_rank, num_procs;
    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    int row_length = a.shape.back();
    int rows_count = a.size() / row_length;

    int base_work = rows_count / num_procs;
    int my_work;
    int start_row, end_row;

    if (my_rank == num_procs - 1)
    {
        my_work = base_work + (rows_count % num_procs);
        start_row = base_work * my_rank;
        end_row = rows_count;
    }
    else
    {
        my_work = base_work;
        start_row = base_work * my_rank;
        end_row = base_work * (my_rank + 1);
    }

    Tensor local_out({my_work, row_length});

    for (int row = 0; row < my_work; row++)
    {
        int offset = row * row_length;
        int global_offset = (start_row + row) * row_length;

        float max_val = a.data[global_offset];
        for (int i = 0; i < row_length; i++)
        {
            max_val = std::max(max_val, a.data[global_offset + i]);
        }

        float sum = 0.f;
        for (int i = 0; i < row_length; i++)
        {
            local_out.data[offset + i] = std::exp(a.data[global_offset + i] - max_val);
            sum += local_out.data[offset + i];
        }

        for (int i = 0; i < row_length; i++)
        {
            local_out.data[offset + i] /= sum;
        }
    }

    Tensor out(a.shape);

    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);

    if (my_rank == ROOT)
    {
        for (int r = 0; r < num_procs; r++)
        {
            int rank_work = base_work;
            if (r == num_procs - 1)
                rank_work = base_work + (rows_count % num_procs);
            recvcounts[r] = rank_work * row_length;
            displs[r] = (base_work * r) * row_length;
        }
    }

    MPI_Gatherv(local_out.data.data(), my_work * row_length, MPI_FLOAT,
                out.data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, ROOT, world);

    MPI_Bcast(out.data.data(), a.size(), MPI_FLOAT, ROOT, world);

    return out;
}
