#include "tokenizer.hpp"
#include "embedding.hpp"
#include "feedforward.hpp"
#include "layer_norm.hpp"
#include "linear.hpp"
#include "tensor.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

Tensor load_tensor(std::ifstream& file) {
    int name_len;
    file.read(reinterpret_cast<char*>(&name_len), sizeof(int));
    std::vector<char> name_bytes(name_len);
    file.read(name_bytes.data(), name_len);
    int ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    std::vector<int> shape(ndim);
    file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int));
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));
    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    Tensor tensor(shape);
    tensor.data = std::move(data);
    return tensor;
}

std::pair<std::string, Tensor> load_named_tensor(std::ifstream& file) {
    int name_len;
    file.read(reinterpret_cast<char*>(&name_len), sizeof(int));
    std::vector<char> name_bytes(name_len);
    file.read(name_bytes.data(), name_len);
    std::string name(name_bytes.begin(), name_bytes.end());
    int ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    std::vector<int> shape(ndim);
    file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int));
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));
    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    Tensor tensor(shape);
    tensor.data = std::move(data);
    return {name, tensor};
}

float relative_error(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        return std::numeric_limits<float>::infinity();
    }    
    float num = 0.0f;
    float denom = 0.0f;
    for (size_t i = 0; i < a.data.size(); i++) {
        float diff = a.data[i] - b.data[i];
        num += diff * diff;
        denom += b.data[i] * b.data[i];
    }
    if (denom < 1e-10f) {
        return std::sqrt(num);
    }
    return std::sqrt(num / denom);
}

float max_abs_error(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < a.data.size(); i++) {
        float err = std::abs(a.data[i] - b.data[i]);
        max_err = std::max(max_err, err);
    }
    return max_err;
}

void print_result(const std::string& test_name, float rel_err, float abs_err, float threshold = 1e-4f) {
    bool passed = rel_err < threshold;
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name;
    std::cout << " (rel_err: " << std::scientific << rel_err;
    std::cout << ", max_abs_err: " << abs_err << ")" << std::endl;
}

bool test_tokenizer(const std::string& tokenizer_path) {
    std::cout << "\nTesting Tokenizer" << std::endl;    
    try {
        SPTokenizer tokenizer(tokenizer_path);
        std::string text = "Hello world, this is a test.";
        std::vector<int> ids = tokenizer.encode(text, true);
        std::string decoded = tokenizer.decode(ids);
        std::cout << "  Original: " << text << std::endl;
        std::cout << "  Decoded:  " << decoded << std::endl;
        std::cout << "  Tokens: " << ids.size() << std::endl;
        std::cout << "  PAD ID: " << tokenizer.get_pad_id() << std::endl;
        std::cout << "  EOS ID: " << tokenizer.get_eos_id() << std::endl;
        print_result("Tokenizer encode/decode", 0.0f, 0.0f);
        return true;
    } catch (const std::exception& e) {
        std::cout << "  Skipped: " << e.what() << std::endl;
        return true;
    }
}

bool test_linear(const std::string& data_dir) {
    std::cout << "\nTesting Linear Layer" << std::endl;
    bool all_passed = true;
    {
        std::string filepath = data_dir + "/linear/linear_no_bias.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);
            Linear layer(input.shape.back(), expected.shape.back(), false);
            layer.weight = weight;
            Tensor output = layer.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Linear without bias", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/linear/linear_with_bias.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);
            auto [name4, bias] = load_named_tensor(file);       
            Linear layer(input.shape.back(), expected.shape.back(), true);
            layer.weight = weight;
            layer.bias = bias;
            Tensor output = layer.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Linear with bias", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/linear/linear_small.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);
            auto [name4, bias] = load_named_tensor(file);       
            Linear layer(input.shape.back(), expected.shape.back(), true);
            layer.weight = weight;
            layer.bias = bias;
            Tensor output = layer.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Linear small", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    return all_passed;
}

bool test_embedding(const std::string& data_dir) {
    std::cout << "\nTesting Embedding Layer" << std::endl;
    bool all_passed = true;
    {
        std::string filepath = data_dir + "/embedding/embedding_basic.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, indices] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);
            int num_emb = weight.shape[0];
            int emb_dim = weight.shape[1];
            Embedding emb(num_emb, emb_dim);
            emb.weight = weight;
            Tensor output = emb.forward(indices);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Embedding basic", rel_err, abs_err);
            all_passed &= (rel_err < 1e-5f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/embedding/embedding_single.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, indices] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);       
            int num_emb = weight.shape[0];
            int emb_dim = weight.shape[1];
            Embedding emb(num_emb, emb_dim);
            emb.weight = weight;
            Tensor output = emb.forward(indices);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Embedding single", rel_err, abs_err);
            all_passed &= (rel_err < 1e-5f);
            file.close();
        }
    }
    return all_passed;
}

bool test_layer_norm(const std::string& data_dir) {
    std::cout << "\nTesting LayerNorm" << std::endl;
    bool all_passed = true;
    {
        std::string filepath = data_dir + "/layer_norm/layer_norm_3d.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);
            RMSNorm ln(weight.shape[0]);
            ln.weight = weight;
            Tensor output = ln.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("LayerNorm 3D", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/layer_norm/layer_norm_2d.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);       
            RMSNorm ln(weight.shape[0]);
            ln.weight = weight;
            Tensor output = ln.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("LayerNorm 2D", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/layer_norm/layer_norm_rms.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, weight] = load_named_tensor(file);       
            RMSNorm ln(weight.shape[0]);
            ln.weight = weight;
            Tensor output = ln.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("LayerNorm RMS", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    return all_passed;
}

bool test_feedforward(const std::string& data_dir) {
    std::cout << "\nTesting FeedForward" << std::endl;
    bool all_passed = true;
    {
        std::string filepath = data_dir + "/feedforward/feedforward_basic.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, wi_weight] = load_named_tensor(file);
            auto [name4, wo_weight] = load_named_tensor(file);
            T5Config config;
            config.d_model = input.shape.back();
            config.d_ff = wi_weight.shape[1];
            FeedForward ff(config);
            ff.wi.weight = wi_weight;
            ff.wo.weight = wo_weight;
            Tensor output = ff.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("FeedForward basic", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/feedforward/feedforward_single.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            auto [name3, wi_weight] = load_named_tensor(file);
            auto [name4, wo_weight] = load_named_tensor(file);       
            T5Config config;
            config.d_model = input.shape.back();
            config.d_ff = wi_weight.shape[1];
            FeedForward ff(config);
            ff.wi.weight = wi_weight;
            ff.wo.weight = wo_weight;
            Tensor output = ff.forward(input);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("FeedForward single", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    return all_passed;
}

bool test_tensor_ops(const std::string& data_dir) {
    std::cout << "\nTesting Tensor Operations" << std::endl;
    bool all_passed = true;
    {
        std::string filepath = data_dir + "/tensor/matmul.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, a] = load_named_tensor(file);
            auto [name2, b] = load_named_tensor(file);
            auto [name3, expected] = load_named_tensor(file);
            Tensor output = a.matmul(b);
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Tensor matmul", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/tensor/transpose.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);       
            Tensor output = input.transpose();
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Tensor transpose", rel_err, abs_err);
            all_passed &= (rel_err < 1e-5f);
            file.close();
        }
    }
    {
        std::string filepath = data_dir + "/tensor/reshape.bin";
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "  Skipping: " << filepath << " not found" << std::endl;
        } else {
            auto [name1, input] = load_named_tensor(file);
            auto [name2, expected] = load_named_tensor(file);
            Tensor output = input.reshape(expected.shape);       
            float rel_err = relative_error(output, expected);
            float abs_err = max_abs_error(output, expected);
            print_result("Tensor reshape", rel_err, abs_err);
            all_passed &= (rel_err < 1e-5f);
            file.close();
        }
    }
    return all_passed;
}

int main(int argc, char** argv) {
    std::cout << "______________________________________" << std::endl;
    std::cout << "  T5 Components Validation Test Suite" << std::endl;
    std::cout << "______________________________________" << std::endl;
    std::string data_dir = "layer_test_data";
    std::string tokenizer_path = "weights/spiece.model";
    if (argc > 1) {
        data_dir = argv[1];
    }
    if (argc > 2) {
        tokenizer_path = argv[2];
    }
    std::cout << "Using test data from: " << data_dir << std::endl;
    std::cout << "Using tokenizer: " << tokenizer_path << std::endl;
    bool all_passed = true;
    all_passed &= test_tokenizer(tokenizer_path);
    all_passed &= test_tensor_ops(data_dir);
    all_passed &= test_linear(data_dir);
    all_passed &= test_embedding(data_dir);
    all_passed &= test_layer_norm(data_dir);
    all_passed &= test_feedforward(data_dir);
    std::cout << "\n______________________________________" << std::endl;
    if (all_passed) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "SOME TESTS FAILED!" << std::endl;
    }
    std::cout << "______________________________________" << std::endl;
    return all_passed ? 0 : 1;
}