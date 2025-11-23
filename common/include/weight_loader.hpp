#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

#include "model.hpp"
#include "tensor.hpp"
#include <fstream>
#include <iostream>
#include <string>

class WeightLoader {
public:
    static bool load_weights(T5Model &model, const std::string &filename);

private:
    static void load_tensor(std::ifstream &file, Tensor &tensor);
    static void load_linear(std::ifstream &file, Linear &linear);
    static void load_block(std::ifstream &file, T5Block &block);
};

#endif 
