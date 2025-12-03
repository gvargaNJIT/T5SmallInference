#include "serial_test_runner.hpp"
#include <stdexcept>

Tensor SerialTestRunner::execute_operation(const std::string& op, const std::vector<Tensor>& inputs) {
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
