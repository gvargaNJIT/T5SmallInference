#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include "embedding.hpp"
#include "tensor.hpp"

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
    
    // Read input (indices as float, need to convert to int)
    test.input = read_tensor(file);
    
    // Read expected output
    test.expected_output = read_tensor(file);
    
    // Read weight
    test.weight = read_tensor(file);
    
    file.close();
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
            std::cout << "Mismatch at index " << i << ": "
                      << "got " << a.data[i] << ", expected " << b.data[i]
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void run_test(const std::string& test_file, int rank) {
    if (rank == 0) {
        std::cout << "Running test: " << test_file << std::endl;
    }
    
    TestCase test;
    
    // Only rank 0 loads the test case
    if (rank == 0) {
        test = load_test_case(test_file);
    }

    // Broadcast dimensions to all ranks
    int num_embeddings, embedding_dim;
    if (rank == 0) {
        num_embeddings = test.weight.shape[0];
        embedding_dim = test.weight.shape[1];
    }
    MPI_Bcast(&num_embeddings, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&embedding_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Create embedding layer on all ranks
    Embedding embedding(num_embeddings, embedding_dim);
    
    // Override weights with test weights on all ranks
    // First broadcast from rank 0
    std::vector<float> weight_buffer(num_embeddings * embedding_dim);
    if (rank == 0) {
        for (int i = 0; i < num_embeddings * embedding_dim; i++) {
            weight_buffer[i] = test.weight.data[i];
        }
    }
    MPI_Bcast(weight_buffer.data(), 
              num_embeddings * embedding_dim, 
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // All ranks copy the broadcasted weights
    for (int i = 0; i < num_embeddings * embedding_dim; i++) {
        embedding.weight.data[i] = weight_buffer[i];
    }
    
    // Broadcast input shape
    int batch, seq_len;
    if (rank == 0) {
        batch = test.input.shape[0];
        seq_len = test.input.shape[1];
    }
    MPI_Bcast(&batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Create input tensor on all ranks if needed
    Tensor input;
    if (rank == 0) {
        input = test.input;
    } else {
        input = Tensor({batch, seq_len});
    }
    
    // *** ADD TIMING HERE - RIGHT BEFORE FORWARD PASS ***
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize all processes
    double start_time = MPI_Wtime();
    
    // Run forward pass (MPI implementation handles distribution internally)
    Tensor output = embedding.forward(input);
    
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize all processes
    double end_time = MPI_Wtime();
    
    // Only rank 0 compares results and reports timing
    if (rank == 0) {
        double elapsed_time = end_time - start_time;
        std::cout << "Time: " << elapsed_time << " seconds" << std::endl;
        
        if (tensors_close(output, test.expected_output)) {
            std::cout << "✓ PASSED" << std::endl;
        } else {
            std::cout << "✗ FAILED" << std::endl;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== MPI Embedding Tests (using " << size << " processes) ===" << std::endl << std::endl;
    }
    
    std::vector<std::string> test_files = {
        "test_cases/embedding/embedding_01.bin",
        "test_cases/embedding/embedding_02.bin",
        "test_cases/embedding/embedding_03.bin",
        "test_cases/embedding/embedding_04.bin",
        "test_cases/embedding/embedding_05.bin"
    };
    
    int passed = 0;
    for (const auto& test_file : test_files) {
        try {
            run_test(test_file, rank);
            if (rank == 0) passed++;
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cout << "✗ FAILED with exception: " << e.what() << std::endl << std::endl;
            }
        }
    }
    
    if (rank == 0) {
        std::cout << "=== Summary ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << test_files.size() << std::endl;
    }
    
    MPI_Finalize();

    if (rank == 0) {
        // Rank 0 returns 0 only if all tests passed
        return (passed == test_files.size()) ? 0 : 1;
    } else {
        // Other ranks finished execution successfully
        return 0;
    }
}