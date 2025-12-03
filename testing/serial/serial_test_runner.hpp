#ifndef SERIAL_TEST_RUNNER_HPP
#define SERIAL_TEST_RUNNER_HPP

#include "../common/test_runner.hpp"

class SerialTestRunner : public TestRunner {
public:
    Tensor execute_operation(const std::string& op, const std::vector<Tensor>& inputs) override;
};

#endif
