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

    Tensor tmp = *this;
    tmp.shape = new_shape;
    return tmp;
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
    int row_length = shape[shape.size() - 1];

    int rows_count = size() / row_length;

    Tensor out(shape);

    for (int row = 0; row < rows_count; row++)
    {
        int offset = row * row_length;

        float max_val = data[offset];
        for (int i = 0; i < row_length; i++)
            max_val = std::max(max_val, data[i+offset]);

        float sum = 0.0f;
        for (int i = 0; i < row_length; i++)
        {
            out.data[offset + i] = std::exp(data[i+offset] - max_val);
            sum += out.data[offset + i];
        }
        
        for (int i = 0; i < row_length; i++)
            out.data[i+offset] /= sum;
    }

    return out;
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
