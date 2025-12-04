#include "tensor.hpp"
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>

extern "C"
{
    Tensor matmul_cuda(const Tensor &a, const Tensor &b);
    Tensor softmax_cuda(const Tensor &input);
}

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<int> &shape_in, float fill_value)
    : shape(shape_in), data(numel(shape_in), fill_value) {}

int Tensor::numel(const std::vector<int> &shape_in)
{
    int total_elements = 1;
    for (int dimension_size : shape_in)
        total_elements *= dimension_size;
    return total_elements;
}

int Tensor::size() const
{
    return static_cast<int>(data.size());
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const
{
    if (numel(new_shape) != size())
        throw std::runtime_error("reshape: size mismatch");

    Tensor tmp = *this;
    tmp.shape = new_shape;
    return tmp;
}

std::vector<int> Tensor::compute_strides(const std::vector<int> &shape_in)
{

    int shape_size = shape_in.size();
    std::vector<int> strides(shape_size);

    int tmp = 1;

    for (int i = shape_size - 1; i >= 0; --i)
    {
        strides[i] = tmp;
        tmp *= shape_in[i];
    }

    return strides;
}

Tensor Tensor::permute(const std::vector<int> &new_axis_order) const
{
    if (new_axis_order.size() != shape.size())
        throw std::runtime_error("permute: wrong dims");

    std::vector<int> permuted_shape(shape.size());
    for (size_t i = 0; i < new_axis_order.size(); i++)
        permuted_shape[i] = shape[new_axis_order[i]];

    Tensor output_tensor(permuted_shape, 0.0f);

    auto input_strides = compute_strides(shape);
    auto output_strides = compute_strides(permuted_shape);

    for (int flat_idx = 0; flat_idx < size(); flat_idx++)
    {
        int remaining_idx = flat_idx;
        std::vector<int> coord(shape.size());

        for (size_t dim = 0; dim < shape.size(); dim++)
        {
            coord[dim] = remaining_idx / input_strides[dim];
            remaining_idx %= input_strides[dim];
        }

        int output_flat_idx = 0;
        for (size_t dim = 0; dim < shape.size(); dim++)
            output_flat_idx += coord[new_axis_order[dim]] * output_strides[dim];

        output_tensor.data[output_flat_idx] = data[flat_idx];
    }

    return output_tensor;
}

Tensor Tensor::transpose() const
{
    if (shape.size() != 2)
        throw std::runtime_error("transpose: only 2D supported");

    return permute({1, 0});
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape != other.shape)
        throw std::runtime_error("operator+: shape mismatch");

    Tensor output_tensor(shape);
    for (int idx = 0; idx < size(); idx++)
        output_tensor.data[idx] = data[idx] + other.data[idx];

    return output_tensor;
}

Tensor Tensor::softmax() const
{
    return softmax_cuda(*this);
}


namespace activation
{
    Tensor relu(const Tensor &input_tensor)
    {
        Tensor output_tensor = input_tensor;

        for (float &value : output_tensor.data)
            value = value > 0 ? value : 0.0f;

        return output_tensor;
    }
}
