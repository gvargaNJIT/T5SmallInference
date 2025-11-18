#ifndef T5_TENSOR_HPP
#define T5_TENSOR_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>

namespace t5 {

class Tensor {
public:
    Tensor() : size_(0) {}
    
    Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_ = 1;
        for (auto dim : shape) {
            size_ *= dim;
        }
        data_.resize(size_, 0.0f);
    }
    
    Tensor(const std::vector<size_t>& shape, float init_value) : shape_(shape) {
        size_ = 1;
        for (auto dim : shape) {
            size_ *= dim;
        }
        data_.resize(size_, init_value);
    }
    
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    size_t size() const { return size_; }
    const std::vector<size_t>& shape() const { return shape_; }

    float& operator[](size_t idx) {
        return data_[idx];
    }
    
    const float& operator[](size_t idx) const {
        return data_[idx];
    }

    float& at(size_t i, size_t j) {
        if (shape_.size() != 2) {
            throw std::runtime_error("at() only works for 2D tensors");
        }
        return data_[i * shape_[1] + j];
    }
    
    const float& at(size_t i, size_t j) const {
        if (shape_.size() != 2) {
            throw std::runtime_error("at() only works for 2D tensors");
        }
        return data_[i * shape_[1] + j];
    }

    float& at(size_t i, size_t j, size_t k) {
        if (shape_.size() != 3) {
            throw std::runtime_error("at() only works for 3D tensors");
        }
        return data_[i * shape_[1] * shape_[2] + j * shape_[2] + k];
    }

    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void zero() {
        fill(0.0f);
    }

    void copy_from(const Tensor& other) {
        if (size_ != other.size_) {
            throw std::runtime_error("Size mismatch in copy_from");
        }
        data_ = other.data_;
    }

    void copy_from(const float* src, size_t count) {
        if (count > size_) {
            throw std::runtime_error("Source data too large");
        }
        std::copy(src, src + count, data_.begin());
    }

    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = 1;
        for (auto dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::runtime_error("Reshape: new size must match old size");
        }
        shape_ = new_shape;
    }

    void print_info() const {
        std::cout << "Tensor shape: [";
        for (size_t i = 0; i < shape_.size(); i++) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], size: " << size_ << std::endl;
    }

    void print_values(size_t max_values = 10) const {
        std::cout << "Values: [";
        for (size_t i = 0; i < std::min(max_values, size_); i++) {
            std::cout << data_[i];
            if (i < std::min(max_values, size_) - 1) std::cout << ", ";
        }
        if (size_ > max_values) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    size_t size_;
};

}
#endif 