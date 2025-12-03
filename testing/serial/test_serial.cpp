#include "serial_test_runner.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::string test_dir = "../test_cases/tensor";
    
    if (argc > 1) {
        test_dir = argv[1];
    }
    
    std::cout << "Serial Tensor Testing Framework" << std::endl;
    std::cout << "Test directory: " << test_dir << std::endl;
    
    SerialTestRunner runner;
    runner.load_test_cases(test_dir);
    runner.run_all_tests();
    
    return (runner.get_failed() > 0) ? 1 : 0;
}
