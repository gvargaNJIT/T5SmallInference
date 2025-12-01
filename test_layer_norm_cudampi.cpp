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
    Tensor input;
    Tensor expected;
    int batch = 0, seq_len = 0, hidden_dim = 0;

    // Root loads test case
    if (rank == 0) {
        test = load_test_case(test_file);
        input = test.input;
        expected = test.expected_output;
        batch = test.input.shape[0];
        seq_len = test.input.shape[1];
        hidden_dim = test.input.shape[2];
    }

    // Broadcast dimensions so all ranks can create properly sized tensors
    MPI_Bcast(&batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hidden_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Non-root ranks need to create input tensor with correct shape
    if (rank != 0) {
        input = Tensor({batch, seq_len, hidden_dim});
    }

    // Create RMSNorm on all ranks
    RMSNorm norm_rms(hidden_dim);

    // Broadcast weights to all ranks
    if (rank == 0) {
        for (int i = 0; i < hidden_dim; i++) {
            norm_rms.weight.data[i] = test.weight.data[i];
        }
    }
    MPI_Bcast(norm_rms.weight.data.data(), hidden_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Call YOUR CUDA+MPI implementation
    // All ranks call this - it handles MPI scatter/gather internally
    Tensor result = norm_rms.forward(input);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Only root checks results
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
        std::cout << "=== MPI+CUDA RMSNorm Tests ===" << std::endl << std::endl;
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
    for (const auto& f : test_files) {
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