#ifndef TEST_RUNNER_HPP
#define TEST_RUNNER_HPP

#include <tensor.hpp>
#include <string>
#include <vector>
#include <functional>

struct TestCase {
    std::string name;
    std::string operation;
    std::vector<Tensor> inputs;
    Tensor expected_output;
};

class TestRunner {
public:
    virtual ~TestRunner() = default;
    
    // Load test cases from directory
    void load_test_cases(const std::string& test_dir);
    
    // Run all tests
    void run_all_tests();
    
    // Run specific test
    bool run_test(const TestCase& test);
    
    // Get results
    int get_passed() const { return passed; }
    int get_failed() const { return failed; }
    
protected:
    virtual Tensor execute_operation(const std::string& op, 
                                     const std::vector<Tensor>& inputs) = 0;
    
    std::vector<TestCase> test_cases;
    
    virtual bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-3f);
    void print_results();
    
    int passed = 0;
    int failed = 0;
};

#endif // TEST_RUNNER_HPP
