#include "feedforward.hpp"
#include "matrix_ops.hpp"
#include <stdexcept>

namespace t5 {
namespace serial {

FeedForward::FeedForward(const Config& config)
    : d_model_(config.d_model)
    , d_ff_(config.d_ff)
    , wi_weight_({d_ff_, d_model_})
    , wo_weight_({d_model_, d_ff_})
{
}

void FeedForward::set_wi_weight(const Tensor& weight) {
    if (weight.shape() != wi_weight_.shape()) {
        throw std::runtime_error("FeedForward: wi_weight shape mismatch");
    }
    wi_weight_.copy_from(weight);
}

void FeedForward::set_wo_weight(const Tensor& weight) {
    if (weight.shape() != wo_weight_.shape()) {
        throw std::runtime_error("FeedForward: wo_weight shape mismatch");
    }
    wo_weight_.copy_from(weight);
}

void FeedForward::forward(const Tensor& input, Tensor& output) {
    auto input_shape = input.shape();
    size_t batch_seq = (input_shape.size() == 3) ? 
        input_shape[0] * input_shape[1] : input_shape[0];
    Tensor input_2d = input;
    if (input_shape.size() == 3) {
        input_2d.reshape({batch_seq, d_model_});
    }
    Tensor hidden({batch_seq, d_ff_});
    matmul_transposed(input_2d.data(), wi_weight_.data(), hidden.data(),
                     batch_seq, d_model_, d_ff_);
    float* hidden_data = hidden.data();
    for (size_t i = 0; i < hidden.size(); i++) {
        if (hidden_data[i] < 0) hidden_data[i] = 0;
    }
    Tensor output_2d({batch_seq, d_model_});
    matmul_transposed(hidden.data(), wo_weight_.data(), output_2d.data(),
                     batch_seq, d_ff_, d_model_);
    if (input_shape.size() == 3) {
        output = Tensor(input_shape);
        output.copy_from(output_2d);
        output.reshape(input_shape);
    } else {
        output = output_2d;
    }
}
}
}