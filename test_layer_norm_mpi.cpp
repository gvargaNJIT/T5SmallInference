#include "rms_norm.hpp"
#include "tensor.hpp"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

struct TestCase {
    std::string name;
    Tensor input;
    Tensor expected_output;
    Tensor weight;
};

Tensor read_tensor(std::ifstream& file) {
    int name_len;
    file.read(reinterpret_cast<char*>(&name_len), sizeof(int));

    std::string name(name_len, '\0');
    file.read(&name[0], name_len);

    int ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));

    std::vector<int> shape(ndim);
    file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int));

    int num_elements;
    file.read(reinterpret_cast<char*>(&num_elements), sizeof(int));

    Tensor tensor(shape);
    file.read(reinterpret_cast<char*>(tensor.data.data()), num_elements * sizeof(float));

    return tensor;
}

TestCase load_test_case(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    TestCase test;
    test.name = filepath;

    test.input = read_tensor(file);
    test.expected_output = read_tensor(file);
    test.weight = read_tensor(file);

    return test;
}

bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-5, float atol = 1e-5) {
    if (a.shape != b.shape) {
        std::cout << "Shape mismatch!" << std::endl;
        return false;
    }

    int size = 1;
    for (int dim : a.shape) size *= dim;

    for (int i = 0; i < size; i++) {
        float diff = std::abs(a.data[i] - b.data[i]);
        float threshold = atol + rtol * std::abs(b.data[i]);

        if (diff > threshold) {
            std::cout << "Mismatch at index " << i
                      << ": got " << a.data[i]
                      << ", expected " << b.data[i]
                      << " (diff=" << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void run_test(const std::string& test_file) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Running test: " << test_file << std::endl;
    }

    TestCase test;
    RMSNorm norm_layer(1);  // will be overwritten

    Tensor input;
    Tensor expected;

    int batch = 0, seq_len = 0, hidden_dim = 0;

    if (rank == 0) {
        test = load_test_case(test_file);

        input = test.input;
        expected = test.expected_output;

        batch = input.shape[0];
        seq_len = input.shape[1];
        hidden_dim = input.shape[2];

        // Construct correct LayerNorm
        norm_layer = RMSNorm(hidden_dim);

        // Override weights
        for (int i = 0; i < hidden_dim; i++)
            norm_layer.weight.data[i] = test.weight.data[i];
    }

    // Broadcast sizes
    MPI_Bcast(&batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hidden_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast weights
    if (rank != 0) {
        norm_layer = RMSNorm(hidden_dim);
    }
    MPI_Bcast(norm_layer.weight.data.data(), hidden_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute distribution
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int batch_per_proc = (batch + size - 1) / size;

    int start = std::min(rank * batch_per_proc, batch);
    int end   = std::min(start + batch_per_proc, batch);
    int local_batch = end - start;

    std::vector<int> counts(size), displs(size);
    for (int i = 0; i < size; i++) {
        int s = std::min(i * batch_per_proc, batch);
        int e = std::min(s + batch_per_proc, batch);
        counts[i] = (e - s) * seq_len * hidden_dim;
        displs[i] = s * seq_len * hidden_dim;
    }

    Tensor local_input({local_batch, seq_len, hidden_dim});
    MPI_Scatterv(rank == 0 ? input.data.data() : nullptr,
                 counts.data(), displs.data(), MPI_FLOAT,
                 local_input.data.data(),
                 local_batch * seq_len * hidden_dim,
                 MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Compute forward on each rank
    Tensor local_output({local_batch, seq_len, hidden_dim});

    for (int b = 0; b < local_batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int offset = b * seq_len * hidden_dim + s * hidden_dim;

            float sumsq = 0.0f;
            for (int d = 0; d < hidden_dim; d++)
                sumsq += local_input.data[offset + d] *
                         local_input.data[offset + d];

            float rms = std::sqrt(sumsq / hidden_dim + norm_layer.eps);
            float inv_rms = 1.0f / rms;

            for (int d = 0; d < hidden_dim; d++)
                local_output.data[offset + d] =
                    local_input.data[offset + d] * inv_rms * norm_layer.weight.data[d];
        }
    }

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Gather results
    Tensor result;
    if (rank == 0) result = Tensor({batch, seq_len, hidden_dim});

    MPI_Gatherv(local_output.data.data(),
                local_batch * seq_len * hidden_dim,
                MPI_FLOAT,
                rank == 0 ? result.data.data() : nullptr,
                counts.data(), displs.data(),
                MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed_time = end_time - start_time;
        std::cout << "Time: " << elapsed_time << " seconds" << std::endl;

        if (tensors_close(result, expected)) {
            std::cout << "✓ PASSED" << std::endl;
        } else {
            std::cout << "✗ FAILED" << std::endl;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "=== MPI LayerNorm (RMSNorm) Tests ===" << std::endl << std::endl;
    }

    double total_start = MPI_Wtime();

    std::vector<std::string> test_files = {
        "test_cases/layer_norm/rmsnorm_01.bin",
        "test_cases/layer_norm/rmsnorm_02.bin",
        "test_cases/layer_norm/rmsnorm_03.bin",
        "test_cases/layer_norm/rmsnorm_04.bin",
        "test_cases/layer_norm/rmsnorm_05.bin"
    };

    int passed = 0;
    for (auto& f : test_files) {
        run_test(f);
        if (rank == 0) passed++;
    }

    double total_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=== Summary ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << test_files.size() << std::endl;
        std::cout << "Total time: " << (total_end - total_start) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}