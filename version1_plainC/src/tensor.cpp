#include "tensor.hpp"
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>

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

    Tensor reshaped_tensor = *this;
    reshaped_tensor.shape = new_shape;
    return reshaped_tensor;
}

std::vector<int> Tensor::compute_strides(const std::vector<int> &shape_in)
{
    std::vector<int> strides(shape_in.size());
    int running_product = 1;

    for (size_t idx = shape_in.size(); idx-- > 0; )
    {
        strides[idx] = running_product;
        running_product *= shape_in[idx];
    }
    return strides;
}

Tensor Tensor::permute(const std::vector<int> &new_axis_order) const
{
    if (new_axis_order.size() != shape.size())
        throw std::runtime_error("permute: wrong dims");

    std::vector<int> permuted_shape(shape.size());
    for (size_t idx = 0; idx < new_axis_order.size(); idx++)
        permuted_shape[idx] = shape[new_axis_order[idx]];

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

Tensor Tensor::softmax(int axis) const
{
    if (axis < 0)
        axis += shape.size();

    Tensor output_tensor = *this;

    int axis_dim_size = shape[axis];

    int inner_stride = 1;
    for (size_t i = axis + 1; i < shape.size(); i++)
        inner_stride *= shape[i];

    int outer_stride = size() / (axis_dim_size * inner_stride);

    for (int outer_idx = 0; outer_idx < outer_stride; outer_idx++)
    {
        for (int inner_idx = 0; inner_idx < inner_stride; inner_idx++)
        {
            float max_value = -1e30f;

            // First pass — find max for numerical stability
            for (int d = 0; d < axis_dim_size; d++)
            {
                int idx =
                    outer_idx * axis_dim_size * inner_stride +
                    d * inner_stride +
                    inner_idx;

                max_value = std::max(max_value, data[idx]);
            }

            float exp_sum = 0.f;

            // Second pass — compute exp(x - max)
            for (int d = 0; d < axis_dim_size; d++)
            {
                int idx =
                    outer_idx * axis_dim_size * inner_stride +
                    d * inner_stride +
                    inner_idx;

                output_tensor.data[idx] = std::exp(data[idx] - max_value);
                exp_sum += output_tensor.data[idx];
            }

            // Third pass — normalize
            for (int d = 0; d < axis_dim_size; d++)
            {
                int idx =
                    outer_idx * axis_dim_size * inner_stride +
                    d * inner_stride +
                    inner_idx;

                output_tensor.data[idx] /= exp_sum;
            }
        }
    }

    return output_tensor;
}

Tensor Tensor::matmul(const Tensor &other) const
{
    int M = shape[0];
    int K_left = shape[1];
    int K_right = other.shape[0];
    int N = other.shape[1];

    if (K_left != K_right)
        throw std::runtime_error("matmul: shape mismatch");

    Tensor output_tensor({M, N});

    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
        {
            float dot_sum = 0.f;

            for (int k = 0; k < K_left; k++)
                dot_sum += data[row * K_left + k] * other.data[k * N + col];

            output_tensor.data[row * N + col] = dot_sum;
        }

    return output_tensor;
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
