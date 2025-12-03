#include "test_runner.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <dirent.h>
#include <cstring>

static int32_t read_int32(std::ifstream &f) {
    int32_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static float read_float32(std::ifstream &f) {
    float v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static Tensor load_one_tensor(std::ifstream &f) {
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

void TestRunner::load_test_cases(const std::string& test_dir) {
    DIR* dir = opendir(test_dir.c_str());
    if (!dir) {
        std::cerr << "Could not open directory: " << test_dir << std::endl;
        return;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        if (filename == "." || filename == "..") continue;
        
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".bin")
            continue;
        
        size_t underscore = filename.find('_');
        if (underscore == std::string::npos) continue;
        
        std::string op = filename.substr(0, underscore);
        
        TestCase tc;
        tc.name = filename;
        tc.operation = op;
        
        // Load the test case file
        std::string filepath = test_dir + "/" + filename;
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Could not open file: " << filepath << std::endl;
            continue;
        }
        
        try {
            if (op == "matmul" || op == "add") {
                tc.inputs.push_back(load_one_tensor(file));
                tc.inputs.push_back(load_one_tensor(file));
                tc.expected_output = load_one_tensor(file);
            } else if (op == "softmax" || op == "permute") {
                tc.inputs.push_back(load_one_tensor(file));
                tc.expected_output = load_one_tensor(file);
            } else {
                std::cerr << "Unknown operation: " << op << std::endl;
                continue;
            }
            
            test_cases.push_back(tc);
        } catch (const std::exception& e) {
            std::cerr << "Error loading " << filename << ": " << e.what() << std::endl;
        }
    }
    
    closedir(dir);
    
    std::cout << "Loaded " << test_cases.size() << " test cases from " << test_dir << std::endl;
}

bool TestRunner::run_test(const TestCase& test) {
    try {
        Tensor result = execute_operation(test.operation, test.inputs);
        
        if (compare_tensors(result, test.expected_output)) {
            std::cout << "✓ " << test.name << " PASSED" << std::endl;
            passed++;
            return true;
        } else {
            std::cout << "✗ " << test.name << " FAILED (mismatch)" << std::endl;
            failed++;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ " << test.name << " ERROR: " << e.what() << std::endl;
        failed++;
        return false;
    }
}

void TestRunner::run_all_tests() {
    std::cout << "\n=== Running Tests ===\n" << std::endl;
    for (const auto& test : test_cases) {
        run_test(test);
    }
    print_results();
}

bool TestRunner::compare_tensors(const Tensor& a, const Tensor& b, float tol) {
    if (a.shape != b.shape) {
        std::cout << "  Shape mismatch!" << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < a.data.size(); i++) {
        float diff = std::fabs(a.data[i] - b.data[i]);
        if (diff > tol) {
            std::cout << "  Value mismatch at index " << i 
                     << ": got " << a.data[i] 
                     << ", expected " << b.data[i] 
                     << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

void TestRunner::print_results() {
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total:  " << (passed + failed) << std::endl;
    
    if (failed == 0) {
        std::cout << "\nAll tests passed!" << std::endl;
    }
}
