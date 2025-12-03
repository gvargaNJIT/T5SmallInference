#pragma once

#include <vector>
#include <functional>

class Tensor {
public:
    std::vector<int> shape;        
    std::vector<float> data;      

    Tensor();
    explicit Tensor(const std::vector<int>& shape_in, float fill_value = 0.0f);

    static int numel(const std::vector<int>& shape_in);

    int size() const;

    Tensor reshape(const std::vector<int> &new_shape) const;

    Tensor permute(const std::vector<int>& new_axis_order) const;

    Tensor transpose() const;

    Tensor softmax() const;

    Tensor matmul(const Tensor& other) const;

    Tensor operator+(const Tensor& other) const;

private:
    static std::vector<int> compute_strides(const std::vector<int>& shape_in);
};

namespace activation {
    Tensor relu(const Tensor& input_tensor);
}

