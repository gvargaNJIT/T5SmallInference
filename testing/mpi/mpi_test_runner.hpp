#ifndef MPI_TEST_RUNNER_HPP
#define MPI_TEST_RUNNER_HPP

#include "../common/test_runner.hpp"
#include <mpi.h>

class MPITestRunner : public TestRunner {
private:
    int rank;
    int size;
    
public:
    MPITestRunner(int r, int s) : rank(r), size(s) {}
    
    Tensor execute_operation(const std::string& op, const std::vector<Tensor>& inputs) override;
    
    void run_all_tests();
    bool run_test(const TestCase& test);
    
protected:
    bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-3f) override;
};

#endif
