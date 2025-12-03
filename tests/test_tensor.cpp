#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>

#include <mpi.h>
#include "tensor.hpp"
#include "mpi_backend.hpp"

// -----------------------------------------------------
// Utility readers
// -----------------------------------------------------
int32_t read_int32(std::ifstream &f) {
    int32_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

float read_float32(std::ifstream &f) {
    float v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

// -----------------------------------------------------
// Load single tensor from binary file
// -----------------------------------------------------
Tensor load_one_tensor(std::ifstream &f) {
    int32_t name_len = read_int32(f);

    std::string name(name_len, '\0');
    f.read(&name[0], name_len);

    int32_t ndim = read_int32(f);
    std::vector<int> shape(ndim);
    for (int i = 0; i < ndim; i++)
        shape[i] = read_int32(f);

    int32_t flat_len = read_int32(f);

    Tensor t(shape);
    if (flat_len != t.size())
        throw std::runtime_error("Flat length mismatch in tensor " + name);

    for (int i = 0; i < flat_len; i++)
        t.data[i] = read_float32(f);

    return t;
}

// -----------------------------------------------------
// Structures for test loading
// -----------------------------------------------------
struct TensorPair {
    Tensor input, output;
};

TensorPair load_tensor_pair(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Could not open " + path);

    return { load_one_tensor(f), load_one_tensor(f) };
}

struct ABOut {
    Tensor a, b, out;
};

ABOut load_ab_out(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Could not open " + path);

    ABOut r;
    r.a = load_one_tensor(f);
    r.b = load_one_tensor(f);
    r.out = load_one_tensor(f);
    return r;
}

// -----------------------------------------------------
// Compare tensors
// -----------------------------------------------------
bool allclose(const Tensor &A, const Tensor &B, float atol = 1e-4f, float rtol = 1e-2f) {
    if (A.shape != B.shape)
        return false;

    for (int i = 0; i < A.size(); i++) {
        float a = A.data[i];
        float b = B.data[i];
        float diff = fabs(a - b);

        float tol = atol + rtol * fabs(b);  // absolute + relative tolerance

        if (diff > tol)
            return false;
    }
    return true;
}


// -----------------------------------------------------
// MAIN TEST DRIVER (with timing)
// -----------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Use absolute test_cases dir when provided by CMake via TEST_CASES_DIR.

    const std::string root = "test_cases/tensor";

    if (rank == 0) {
        std::cout << "=== Testing Tensor (serial + MPI backend) ===\n";
        std::cout << "World size: " << world << "\n";
    }

    // =====================================================
    // 1. PERMUTE (serial only)
    // =====================================================
    for (int i = 1; i <= 5; i++) {
        std::string path = root + "/permute_" + std::to_string(i) + ".bin";

        TensorPair pair = load_tensor_pair(path);

        Tensor inp = pair.input;
        Tensor ref = pair.output;

        // Infer permutation
        std::vector<int> perm(inp.shape.size());
        for (int p = 0; p < (int)perm.size(); p++)
            for (int q = 0; q < (int)perm.size(); q++)
                if (ref.shape[p] == inp.shape[q]) {
                    perm[p] = q;
                    break;
                }

        double t0 = MPI_Wtime();
        Tensor out = inp.permute(perm);
        double t1 = MPI_Wtime();

        bool ok = allclose(out, ref);

        if (rank == 0)
            std::cout << "permute_" << i << " [serial]: "
                      << (ok ? "PASS" : "FAIL")
                      << " (" << (t1 - t0) * 1000.0 << " ms)\n";
    }

    // =====================================================
    // 2. SOFTMAX (serial + MPI)
    // =====================================================
    for (int i = 1; i <= 5; i++) {
        std::string path = root + "/softmax_" + std::to_string(i) + ".bin";
        TensorPair pair = load_tensor_pair(path);

        Tensor inp = pair.input;
        Tensor ref = pair.output;
        int axis = (int)inp.shape.size() - 1;

        // ---- Serial ----
        double t0s = MPI_Wtime();
        Tensor out_serial = inp.softmax();
        double t1s = MPI_Wtime();

        bool ok_serial = allclose(out_serial, ref);

        if (rank == 0)
            std::cout << "softmax_" << i << " [serial]: "
                      << (ok_serial ? "PASS" : "FAIL")
                      << " (" << (t1s - t0s) * 1000.0 << " ms)\n";

        // ---- MPI ----
        MPI_Barrier(MPI_COMM_WORLD);
        double t0m = MPI_Wtime();
        Tensor out_mpi = mpi_backend::softmax(inp);
        double t1m = MPI_Wtime();

        bool ok_mpi = true;
        if (rank == 0)
            ok_mpi = allclose(out_mpi, ref);

        if (rank == 0)
            std::cout << "softmax_" << i << " [mpi_backend]: "
                      << (ok_mpi ? "PASS" : "FAIL")
                      << " (" << (t1m - t0m) * 1000.0 << " ms)\n";
    }

    // =====================================================
    // 3. MATMUL (serial + MPI)
    // =====================================================
    for (int i = 1; i <= 5; i++) {
        std::string path = root + "/matmul_" + std::to_string(i) + ".bin";
        ABOut pack = load_ab_out(path);

        // ---- Serial ----
        double t0s = MPI_Wtime();
        Tensor out_serial = pack.a.matmul(pack.b);
        double t1s = MPI_Wtime();
// 
        bool ok_serial = allclose(out_serial, pack.out);

        if (rank == 0)
            std::cout << "matmul_" << i << " [serial]: "
                      << (ok_serial ? "PASS" : "FAIL")
                      << " (" << (t1s - t0s) * 1000.0 << " ms)\n";

        // ---- MPI ----
        MPI_Barrier(MPI_COMM_WORLD);
        double t0m = MPI_Wtime();
        Tensor out_mpi = mpi_backend::matmul(pack.a, pack.b);
        double t1m = MPI_Wtime();

        bool ok_mpi = true;
        if (rank == 0)
            ok_mpi = allclose(out_mpi, pack.out);

        if (rank == 0)
            std::cout << "matmul_" << i << " [mpi_backend]: "
                      << (ok_mpi ? "PASS" : "FAIL")
                      << " (" << (t1m - t0m) * 1000.0 << " ms)\n";
    }

    // =====================================================
    // 4. ADD (serial only)
    // =====================================================
    for (int i = 1; i <= 5; i++) {
        std::string path = root + "/add_" + std::to_string(i) + ".bin";
        ABOut pack = load_ab_out(path);

        double t0 = MPI_Wtime();
        Tensor out = pack.a + pack.b;
        double t1 = MPI_Wtime();

        bool ok = allclose(out, pack.out);

        if (rank == 0)
            std::cout << "add_" << i << " [serial]: "
                      << (ok ? "PASS" : "FAIL")
                      << " (" << (t1 - t0) * 1000.0 << " ms)\n";
    }

    if (rank == 0)
        std::cout << "\nAll tensor tests complete.\n";

    MPI_Finalize();
    return 0;
}
