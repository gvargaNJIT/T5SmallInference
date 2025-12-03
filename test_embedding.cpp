#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include "embedding.hpp"
#include "tensor.hpp"
#include <chrono>

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
    
    // Read input (indices as float, need to keep as-is for now)
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

void run_test(const std::string& test_file) {
    std::cout << "Running test: " << test_file << std::endl;
    
    TestCase test = load_test_case(test_file);
    
    // Create embedding layer with matching dimensions
    int num_embeddings = test.weight.shape[0];
    int embedding_dim = test.weight.shape[1];
    
    Embedding embedding(num_embeddings, embedding_dim);
    
    // Override with test weights
    int weight_size = num_embeddings * embedding_dim;
    for (int i = 0; i < weight_size; i++) {
        embedding.weight.data[i] = test.weight.data[i];
    }
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run forward pass
    Tensor output = embedding.forward(test.input);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double elapsed_seconds = elapsed.count() / 1000000.0;
    
    std::cout << "Time: " << elapsed_seconds << " seconds" << std::endl;

    // Compare results
    if (tensors_close(output, test.expected_output)) {
        std::cout << "✓ PASSED" << std::endl;
    } else {
        std::cout << "✗ FAILED" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Embedding Tests ===" << std::endl << std::endl;
    
    // Start total timing
    auto total_start = std::chrono::high_resolution_clock::now();
    
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
            run_test(test_file);
            passed++;
        } catch (const std::exception& e) {
            std::cout << "✗ FAILED with exception: " << e.what() << std::endl << std::endl;
        }
    }
    
    // End total timing
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    double total_seconds = total_elapsed.count() / 1000000.0;
    
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << test_files.size() << std::endl;
    std::cout << "Total time: " << total_seconds << " seconds" << std::endl;
    
    return (passed == test_files.size()) ? 0 : 1;
}