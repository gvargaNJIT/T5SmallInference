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

Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape != other.shape)
        throw std::runtime_error("operator+: shape mismatch");

    Tensor output_tensor(shape);
    for (int idx = 0; idx < size(); idx++)
        output_tensor.data[idx] = data[idx] + other.data[idx];

    return output_tensor;
}


Tensor Tensor::permute(const std::vector<int> &axes) const
{
    int dims = shape.size();
    if (axes.size() != dims && dims != 3)
        throw std::runtime_error("permute: wrong dims and only 3D supported");

    int A = shape[0];
    int B = shape[1];
    int C = shape[2];

    int sA = shape[axes[0]];
    int sB = shape[axes[1]];
    int sC = shape[axes[2]];

    Tensor out({sA, sB, sC}, 0.f);

    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < B; j++)
        {
            for (int k = 0; k < C; k++)
            {
                int tmp[3] = {i, j, k};
                int ai = tmp[axes[0]];
                int bi = tmp[axes[1]];
                int ci = tmp[axes[2]];

                int in_flat = i * B * C + j * C + k;
                int out_flat = ai * (sB * sC) + bi * sC + ci;

                out.data[out_flat] = data[in_flat];
            }
        }
    }

    return out;
}

Tensor Tensor::transpose() const
{
    if (shape.size() != 2)
        throw std::runtime_error("transpose: only 2D supported");

    int H = shape[0];
    int W = shape[1];

    Tensor out({W, H}, 0.f);

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            out.data[j * H + i] = data[i * W + j];

    return out;
}


Tensor Tensor::softmax() const
{
    return softmax_cuda(*this);
}

Tensor Tensor::matmul(const Tensor &other) const
{
    if (shape[1] != other.shape[0])
        throw std::runtime_error("matmul: shape mismatch");

    return matmul_cuda(*this, other);
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
