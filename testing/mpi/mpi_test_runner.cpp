#include "mpi_test_runner.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

Tensor MPITestRunner::execute_operation(const std::string& op, const std::vector<Tensor>& inputs) {
    if (op == "matmul") {
        if (inputs.size() != 2)
            throw std::runtime_error("matmul requires 2 inputs");
        return inputs[0].matmul(inputs[1]);
    }
    else if (op == "add") {
        if (inputs.size() != 2)
            throw std::runtime_error("add requires 2 inputs");
        return inputs[0] + inputs[1];
    }
    else if (op == "softmax") {
        if (inputs.size() != 1)
            throw std::runtime_error("softmax requires 1 input");
        return inputs[0].softmax();
    }
    else {
        throw std::runtime_error("Unknown operation: " + op);
    }
}

void MPITestRunner::run_all_tests() {
    if (rank == 0) {
        std::cout << "\n=== Running MPI Tests (ranks=" << size << ") ===\n" << std::endl;
    }
    
    for (const auto& test : test_cases) {
        run_test(test);
    }
    
    if (rank == 0) {
        print_results();
    }
}

bool MPITestRunner::run_test(const TestCase& test) {
    try {
        Tensor result = execute_operation(test.operation, test.inputs);
        
        bool success = compare_tensors(result, test.expected_output);
        
        if (rank == 0) {
            if (success) {
                std::cout << "✓ " << test.name << " PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "✗ " << test.name << " FAILED (mismatch)" << std::endl;
                failed++;
            }
        }
        
        return success;
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cout << "✗ " << test.name << " ERROR: " << e.what() << std::endl;
            failed++;
        }
        return false;
    }
}

bool MPITestRunner::compare_tensors(const Tensor& a, const Tensor& b, float tol) {
    if (a.shape != b.shape) {
        if (rank == 0) {
            std::cout << "  Shape mismatch!" << std::endl;
        }
        return false;
    }
    
    for (size_t i = 0; i < a.data.size(); i++) {
        float diff = std::fabs(a.data[i] - b.data[i]);
        if (diff > tol) {
            if (rank == 0) {
                std::cout << "  Value mismatch at index " << i 
                         << ": got " << a.data[i] 
                         << ", expected " << b.data[i] 
                         << ", diff=" << diff << std::endl;
            }
            return false;
        }
    }
    return true;
}
