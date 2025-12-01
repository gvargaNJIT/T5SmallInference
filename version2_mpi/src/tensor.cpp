#include <mpi.h>
#include <stdexcept>
#include <algorithm>
#include "mpi_backend.hpp"
#include <cmath>

namespace mpi_backend {

Tensor matmul(const Tensor& A, const Tensor& B)
{
    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M  = A.shape[0];
    int K1 = A.shape[1];
    int K2 = B.shape[0];
    int N  = B.shape[1];

    if (K1 != K2)
        throw std::runtime_error("shape mismatch");

    // -----------------------------------------------------
    // 1. Broadcast B (best done with collectives)
    // -----------------------------------------------------
    MPI_Bcast((void*)B.data.data(), B.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------
    // 2. Compute row partition
    // -----------------------------------------------------
    std::vector<int> rows(world), row_displs(world);

    int base = M / world;
    int rem  = M % world;

    int offset = 0;
    for (int r = 0; r < world; r++) {
        rows[r] = base + (r < rem ? 1 : 0);
        row_displs[r] = offset;
        offset += rows[r];
    }

    int local_rows = rows[rank];

    // -----------------------------------------------------
    // 3. Create RMA windows
    // -----------------------------------------------------
    // Root holds the entire output (M × N)
    Tensor C({M, N});
    float* C_buf = (rank == 0 ? C.data.data() : nullptr);

    MPI_Win winA, winC;

    // A window: root exposes A so workers can GET rows
    MPI_Win_create(
        (void*)(rank == 0 ? A.data.data() : nullptr),
        (rank == 0 ? A.size() * sizeof(float) : 0),
        sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winA);

    // C window: root exposes output buffer so workers PUT results
    MPI_Win_create(
        (void*)C_buf,
        (rank == 0 ? C.size() * sizeof(float) : 0),
        sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winC);

    // -----------------------------------------------------
    // 4. Each rank GETs its A rows from root
    // -----------------------------------------------------
    Tensor A_local({local_rows, K1});

    MPI_Win_fence(0, winA);

    MPI_Get(
        A_local.data.data(),              // local buffer
        local_rows * K1, MPI_FLOAT,
        0,                                // get from root
        row_displs[rank] * K1,            // displacement in floats
        local_rows * K1, MPI_FLOAT,
        winA
    );

    MPI_Win_fence(0, winA);
    MPI_Win_free(&winA);

    // -----------------------------------------------------
    // 5. Local compute: C_local = A_local × B
    // -----------------------------------------------------
    Tensor C_local({local_rows, N}, 0.0f);

    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < K1; k++)
                sum += A_local.data[i*K1 + k] * B.data[k*N + j];
            C_local.data[i*N + j] = sum;
        }

    // -----------------------------------------------------
    // 6. PUT results back into root buffer
    // -----------------------------------------------------
    MPI_Win_fence(0, winC);

    if (rank != 0) {
        MPI_Put(
            C_local.data.data(),
            local_rows * N, MPI_FLOAT,
            0,                                // PUT into root
            row_displs[rank] * N,             // displacement in floats
            local_rows * N, MPI_FLOAT,
            winC
        );
    }
    else {
        // root copies its own part
        for (int i = 0; i < local_rows * N; i++)
            C.data[i] = C_local.data[i];
    }

    MPI_Win_fence(0, winC);
    MPI_Win_free(&winC);

    return C;
}


Tensor softmax(const Tensor& X)
{
    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // -----------------------------------------------------
    // 1. Flatten everything except last dim
    // -----------------------------------------------------
    int ndim = X.shape.size();
    int dim  = X.shape[ndim - 1];

    int rows = 1;
    for (int i = 0; i < ndim - 1; i++)
        rows *= X.shape[i];

    // -----------------------------------------------------
    // 2. Row partition
    // -----------------------------------------------------
    int base = rows / world;
    int rem  = rows % world;

    int local_rows = base + (rank < rem ? 1 : 0);
    int start_row  = rank * base + std::min(rank, rem);

    // -----------------------------------------------------
    // 3. Create windows with passive-target RMA
    // -----------------------------------------------------
    float* root_X = nullptr;
    if (rank == 0) root_X = const_cast<float*>(X.data.data());

    MPI_Win winX;
    MPI_Win_create(
        (void*)root_X,
        rank == 0 ? X.size()*sizeof(float) : 0,
        sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winX
    );

    Tensor final_flat({rows, dim});
    float* root_out = nullptr;
    if (rank == 0) root_out = final_flat.data.data();

    MPI_Win winOut;
    MPI_Win_create(
        (void*)root_out,
        rank == 0 ? final_flat.size()*sizeof(float) : 0,
        sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winOut
    );

    // -----------------------------------------------------
    // 4. GET data (no fences)
    // -----------------------------------------------------
    Tensor X_local({local_rows, dim});

    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, winX);

    MPI_Get(
        X_local.data.data(),
        local_rows * dim, MPI_FLOAT,
        0,
        start_row * dim,
        local_rows * dim,
        MPI_FLOAT,
        winX
    );

    MPI_Win_unlock(0, winX);
    MPI_Win_free(&winX);

    // -----------------------------------------------------
    // 5. Compute softmax
    // -----------------------------------------------------
    Tensor local_out({local_rows, dim});

    for (int lr = 0; lr < local_rows; lr++) {
        float maxv = -1e30f;
        for (int j = 0; j < dim; j++)
            maxv = std::max(maxv, X_local.data[lr*dim + j]);

        float sum = 0.f;
        for (int j = 0; j < dim; j++) {
            float e = std::exp(X_local.data[lr*dim + j] - maxv);
            local_out.data[lr*dim + j] = e;
            sum += e;
        }
        for (int j = 0; j < dim; j++)
            local_out.data[lr*dim + j] /= sum;
    }

    // -----------------------------------------------------
    // 6. PUT results back to root (no fences)
    // -----------------------------------------------------
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, winOut);

    if (rank != 0) {
        MPI_Put(
            local_out.data.data(),
            local_rows * dim, MPI_FLOAT,
            0,
            start_row * dim,
            local_rows * dim,
            MPI_FLOAT,
            winOut
        );
    } else {
        // root writes its own rows
        for (int i = 0; i < local_rows * dim; i++)
            final_flat.data[i] = local_out.data[i];
    }

    MPI_Win_unlock(0, winOut);
    MPI_Win_free(&winOut);

    // -----------------------------------------------------
    // 7. Root reshapes output
    // -----------------------------------------------------
    Tensor final_output(X.shape);

    if (rank == 0) {
        for (int i = 0; i < rows * dim; i++)
            final_output.data[i] = final_flat.data[i];
    }

    return final_output;
}


} // namespace mpi_backend
