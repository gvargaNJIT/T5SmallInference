#include "mpi_test_runner.hpp"
#include <iostream>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::string test_dir = "../test_cases/tensor";
    
    if (argc > 1) {
        test_dir = argv[1];
    }
    
    if (rank == 0) {
        std::cout << "MPI Tensor Testing Framework" << std::endl;
        std::cout << "Test directory: " << test_dir << std::endl;
        std::cout << "Running with " << size << " MPI processes" << std::endl;
    }
    
    MPITestRunner runner(rank, size);
    
    // All ranks load test cases (they all need to know what to execute)
    runner.load_test_cases(test_dir);
    
    runner.run_all_tests();
    
    int exit_code = (runner.get_failed() > 0) ? 1 : 0;
    
    MPI_Finalize();
    return exit_code;
}
