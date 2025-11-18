#ifndef T5_WEIGHT_LOADER_HPP
#define T5_WEIGHT_LOADER_HPP

#include "tensor.hpp"
#include <string>
#include <unordered_map>

namespace t5 {

class WeightLoader {
public:
    static std::unordered_map<std::string, Tensor> load_weights(
        const std::string& weights_dir
    );

    static void print_weight_info(
        const std::unordered_map<std::string, Tensor>& weights
    );
    
private:
    static Tensor load_single_weight(
        const std::string& bin_file,
        const std::string& shape_file
    );

    static std::vector<size_t> parse_shape_file(const std::string& shape_file);
};

}
#endif