#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>

// Simple Tensor class for pure C++ implementation
class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    // Constructors
    Tensor() = default;
    
    Tensor(const std::vector<int>& shape) : shape(shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, 0.0f);
    }

    Tensor(const std::vector<int>& shape, float init_value) : shape(shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, init_value);
    }

    // Static factory methods
    static Tensor zeros(const std::vector<int>& shape) {
        return Tensor(shape, 0.0f);
    }

    static Tensor ones(const std::vector<int>& shape) {
        return Tensor(shape, 1.0f);
    }

    static Tensor randn(const std::vector<int>& shape, float mean = 0.0f, float stddev = 1.0f) {
        Tensor t(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, stddev);
        
        for (auto& val : t.data) {
            val = dist(gen);
        }
        return t;
    }

    static Tensor rand_uniform(const std::vector<int>& shape, float low = 0.0f, float high = 1.0f) {
        Tensor t(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(low, high);
        
        for (auto& val : t.data) {
            val = dist(gen);
        }
        return t;
    }

    // Accessors
    int size() const {
        return data.size();
    }

    int numel() const {
        return data.size();
    }

    int ndim() const {
        return shape.size();
    }

    // Get element at multi-dimensional index
    float& at(const std::vector<int>& indices) {
        int flat_idx = compute_flat_index(indices);
        return data[flat_idx];
    }

    const float& at(const std::vector<int>& indices) const {
        int flat_idx = compute_flat_index(indices);
        return data[flat_idx];
    }

    // Reshape
    Tensor reshape(const std::vector<int>& new_shape) const {
        int new_size = 1;
        for (int dim : new_shape) new_size *= dim;
        
        if (new_size != size()) {
            throw std::runtime_error("Reshape: size mismatch");
        }
        
        Tensor result;
        result.shape = new_shape;
        result.data = data;
        return result;
    }

    // Transpose (for 2D tensors)
    Tensor transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Transpose only supports 2D tensors");
        }
        
        int rows = shape[0];
        int cols = shape[1];
        Tensor result({cols, rows});
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j * rows + i] = data[i * cols + j];
            }
        }
        
        return result;
    }

    // Transpose with specific dimensions
    Tensor permute(const std::vector<int>& dims) const {
        if (dims.size() != shape.size()) {
            throw std::runtime_error("Permute: dims size must match tensor dimensions");
        }
        
        // Handle 4D permutation [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        if (shape.size() == 4 && dims.size() == 4) {
            std::vector<int> new_shape(4);
            for (int i = 0; i < 4; i++) {
                new_shape[i] = shape[dims[i]];
            }
            
            Tensor result(new_shape);
            
            // Compute strides for source and destination
            std::vector<int> src_strides(4);
            std::vector<int> dst_strides(4);
            
            src_strides[3] = 1;
            src_strides[2] = shape[3];
            src_strides[1] = shape[2] * src_strides[2];
            src_strides[0] = shape[1] * src_strides[1];
            
            dst_strides[3] = 1;
            dst_strides[2] = new_shape[3];
            dst_strides[1] = new_shape[2] * dst_strides[2];
            dst_strides[0] = new_shape[1] * dst_strides[1];
            
            // Permute data
            int total_elements = data.size();
            for (int i = 0; i < total_elements; i++) {
                // Compute multi-dimensional index in source
                int idx0 = i / src_strides[0];
                int rem = i % src_strides[0];
                int idx1 = rem / src_strides[1];
                rem = rem % src_strides[1];
                int idx2 = rem / src_strides[2];
                int idx3 = rem % src_strides[2];
                
                std::vector<int> src_indices = {idx0, idx1, idx2, idx3};
                
                // Map to destination indices
                std::vector<int> dst_indices(4);
                for (int j = 0; j < 4; j++) {
                    dst_indices[j] = src_indices[dims[j]];
                }
                
                // Compute flat destination index
                int dst_idx = dst_indices[0] * dst_strides[0] + 
                             dst_indices[1] * dst_strides[1] + 
                             dst_indices[2] * dst_strides[2] + 
                             dst_indices[3] * dst_strides[3];
                
                result.data[dst_idx] = data[i];
            }
            
            return result;
        }
        
        // Handle 2D transpose
        if (shape.size() == 2 && dims == std::vector<int>{1, 0}) {
            return transpose();
        }
        
        throw std::runtime_error("Permute: unsupported dimension order or tensor shape");
    }

    // View (reshape without copy)
    Tensor view(const std::vector<int>& new_shape) const {
        return reshape(new_shape);
    }

    // Slice along first dimension
    Tensor slice(int start, int end) const {
        if (shape.empty()) throw std::runtime_error("Cannot slice scalar");
        
        int first_dim = shape[0];
        if (end == -1) end = first_dim;
        
        int slice_size = 1;
        for (size_t i = 1; i < shape.size(); i++) {
            slice_size *= shape[i];
        }
        
        std::vector<int> new_shape = shape;
        new_shape[0] = end - start;
        
        Tensor result(new_shape);
        std::copy(data.begin() + start * slice_size,
                  data.begin() + end * slice_size,
                  result.data.begin());
        
        return result;
    }

    // Unsqueeze (add dimension)
    Tensor unsqueeze(int dim) const {
        std::vector<int> new_shape = shape;
        if (dim < 0) dim = shape.size() + dim + 1;
        new_shape.insert(new_shape.begin() + dim, 1);
        return reshape(new_shape);
    }

    // Squeeze (remove dimensions of size 1)
    Tensor squeeze(int dim = -1) const {
        std::vector<int> new_shape;
        for (size_t i = 0; i < shape.size(); i++) {
            if (dim >= 0) {
                if ((int)i == dim && shape[i] == 1) continue;
                new_shape.push_back(shape[i]);
            } else {
                if (shape[i] != 1) new_shape.push_back(shape[i]);
            }
        }
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);
    }

    // Math operations
    Tensor operator+(const Tensor& other) const {
        return elementwise_op(other, [](float a, float b) { return a + b; });
    }

    Tensor operator-(const Tensor& other) const {
        return elementwise_op(other, [](float a, float b) { return a - b; });
    }

    Tensor operator*(const Tensor& other) const {
        return elementwise_op(other, [](float a, float b) { return a * b; });
    }

    Tensor operator/(const Tensor& other) const {
        return elementwise_op(other, [](float a, float b) { return a / b; });
    }

    Tensor operator+(float scalar) const {
        return scalar_op(scalar, [](float a, float b) { return a + b; });
    }

    Tensor operator*(float scalar) const {
        return scalar_op(scalar, [](float a, float b) { return a * b; });
    }

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const {
        if (shape.size() < 2 || other.shape.size() < 2) {
            throw std::runtime_error("Matmul requires at least 2D tensors");
        }

        // Simple 2D matmul
        if (shape.size() == 2 && other.shape.size() == 2) {
            return matmul_2d(other);
        }

        // Batched matmul
        return batched_matmul(other);
    }

    // Power
    Tensor pow(float exponent) const {
        Tensor result = *this;
        for (auto& val : result.data) {
            val = std::pow(val, exponent);
        }
        return result;
    }

    // Square root
    Tensor sqrt() const {
        Tensor result = *this;
        for (auto& val : result.data) {
            val = std::sqrt(val);
        }
        return result;
    }

    // Reciprocal square root
    Tensor rsqrt() const {
        Tensor result = *this;
        for (auto& val : result.data) {
            val = 1.0f / std::sqrt(val);
        }
        return result;
    }

    // Mean along dimension
    Tensor mean(int dim, bool keepdim = false) const {
        if (dim < 0) dim = shape.size() + dim;
        
        std::vector<int> new_shape;
        for (size_t i = 0; i < shape.size(); i++) {
            if ((int)i == dim) {
                if (keepdim) new_shape.push_back(1);
            } else {
                new_shape.push_back(shape[i]);
            }
        }
        if (new_shape.empty()) new_shape.push_back(1);
        
        Tensor result(new_shape);
        
        // Compute mean
        int dim_size = shape[dim];
        int outer_size = 1;
        for (int i = 0; i < dim; i++) outer_size *= shape[i];
        
        int inner_size = 1;
        for (size_t i = dim + 1; i < shape.size(); i++) inner_size *= shape[i];
        
        for (int outer = 0; outer < outer_size; outer++) {
            for (int inner = 0; inner < inner_size; inner++) {
                float sum = 0.0f;
                for (int d = 0; d < dim_size; d++) {
                    int idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                    sum += data[idx];
                }
                int result_idx = outer * inner_size + inner;
                result.data[result_idx] = sum / dim_size;
            }
        }
        
        return result;
    }

    // Softmax
    Tensor softmax(int dim) const {
        if (dim < 0) dim = shape.size() + dim;
        
        Tensor result = *this;
        
        int dim_size = shape[dim];
        int outer_size = 1;
        for (int i = 0; i < dim; i++) outer_size *= shape[i];
        
        int inner_size = 1;
        for (size_t i = dim + 1; i < shape.size(); i++) inner_size *= shape[i];
        
        for (int outer = 0; outer < outer_size; outer++) {
            for (int inner = 0; inner < inner_size; inner++) {
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (int d = 0; d < dim_size; d++) {
                    int idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                    max_val = std::max(max_val, data[idx]);
                }
                
                // Compute exp and sum
                float sum = 0.0f;
                for (int d = 0; d < dim_size; d++) {
                    int idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                    result.data[idx] = std::exp(data[idx] - max_val);
                    sum += result.data[idx];
                }
                
                // Normalize
                for (int d = 0; d < dim_size; d++) {
                    int idx = outer * (dim_size * inner_size) + d * inner_size + inner;
                    result.data[idx] /= sum;
                }
            }
        }
        
        return result;
    }

    // Concatenate
    static Tensor cat(const std::vector<Tensor>& tensors, int dim) {
        if (tensors.empty()) throw std::runtime_error("Cannot concatenate empty list");
        
        // Calculate new shape
        std::vector<int> new_shape = tensors[0].shape;
        for (size_t i = 1; i < tensors.size(); i++) {
            new_shape[dim] += tensors[i].shape[dim];
        }
        
        Tensor result(new_shape);
        
        // Simple case: concatenate along first dimension
        if (dim == 0) {
            int offset = 0;
            for (const auto& t : tensors) {
                std::copy(t.data.begin(), t.data.end(), result.data.begin() + offset);
                offset += t.data.size();
            }
        } else if (dim == 1 && tensors[0].shape.size() == 2) {
            // Concatenate along columns
            int rows = tensors[0].shape[0];
            int offset = 0;
            
            for (int r = 0; r < rows; r++) {
                offset = 0;
                for (const auto& t : tensors) {
                    int cols = t.shape[1];
                    for (int c = 0; c < cols; c++) {
                        result.data[r * new_shape[1] + offset + c] = t.data[r * cols + c];
                    }
                    offset += cols;
                }
            }
        }
        
        return result;
    }

private:
    int compute_flat_index(const std::vector<int>& indices) const {
        int flat_idx = 0;
        int multiplier = 1;
        
        for (int i = shape.size() - 1; i >= 0; i--) {
            flat_idx += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        
        return flat_idx;
    }

    Tensor elementwise_op(const Tensor& other, std::function<float(float, float)> op) const {
        // Broadcasting support
        if (shape == other.shape) {
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = op(data[i], other.data[i]);
            }
            return result;
        }
        
        // Simple broadcasting for common cases
        if (other.size() == 1) {
            return scalar_op(other.data[0], op);
        }
        
        throw std::runtime_error("Broadcasting not fully supported");
    }

    Tensor scalar_op(float scalar, std::function<float(float, float)> op) const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = op(data[i], scalar);
        }
        return result;
    }

    Tensor matmul_2d(const Tensor& other) const {
        int m = shape[0];
        int k = shape[1];
        int n = other.shape[1];
        
        if (k != other.shape[0]) {
            throw std::runtime_error("Matmul dimension mismatch");
        }
        
        Tensor result({m, n});
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += data[i * k + p] * other.data[p * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }
        
        return result;
    }

    Tensor batched_matmul(const Tensor& other) const {
        // For 3D and 4D batched matmul
        std::vector<int> batch_dims_this;
        std::vector<int> batch_dims_other;
        
        for (size_t i = 0; i < shape.size() - 2; i++) {
            batch_dims_this.push_back(shape[i]);
        }
        for (size_t i = 0; i < other.shape.size() - 2; i++) {
            batch_dims_other.push_back(other.shape[i]);
        }
        
        // Calculate batch size
        int batch_size = 1;
        for (int dim : batch_dims_this) batch_size *= dim;
        
        int m = shape[shape.size() - 2];
        int k = shape[shape.size() - 1];
        int n = other.shape[other.shape.size() - 1];
        
        std::vector<int> result_shape = batch_dims_this;
        result_shape.push_back(m);
        result_shape.push_back(n);
        
        Tensor result(result_shape);
        
        int mk_size = m * k;
        int kn_size = k * n;
        int mn_size = m * n;
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += data[b * mk_size + i * k + p] * 
                               other.data[b * kn_size + p * n + j];
                    }
                    result.data[b * mn_size + i * n + j] = sum;
                }
            }
        }
        
        return result;
    }
};

// Activation functions
namespace activation {
    inline Tensor relu(const Tensor& x) {
        Tensor result = x;
        for (auto& val : result.data) {
            val = std::max(0.0f, val);
        }
        return result;
    }

    inline Tensor gelu(const Tensor& x) {
        Tensor result = x;
        for (auto& val : result.data) {
            val = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * std::pow(val, 3.0f))));
        }
        return result;
    }

    inline Tensor tanh(const Tensor& x) {
        Tensor result = x;
        for (auto& val : result.data) {
            val = std::tanh(val);
        }
        return result;
    }
}

#endif // TENSOR_HPP