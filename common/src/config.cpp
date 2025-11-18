
#include "config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace t5 {

Config Config::load(const std::string& filepath) {
    Config config;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open config file: " << filepath << std::endl;
        std::cerr << "Using default T5-small configuration" << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\"'), line.end());
        line.erase(std::remove(line.begin(), line.end(), ','), line.end());

        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        
        std::string key = line.substr(0, colon);
        std::string value = line.substr(colon + 1);

        if (key == "d_model") {
            config.d_model = std::stoul(value);
        } else if (key == "d_kv") {
            config.d_kv = std::stoul(value);
        } else if (key == "d_ff") {
            config.d_ff = std::stoul(value);
        } else if (key == "num_layers") {
            config.num_layers = std::stoul(value);
        } else if (key == "num_decoder_layers") {
            config.num_decoder_layers = std::stoul(value);
        } else if (key == "num_heads") {
            config.num_heads = std::stoul(value);
        } else if (key == "vocab_size") {
            config.vocab_size = std::stoul(value);
        } else if (key == "pad_token_id") {
            config.pad_token_id = std::stoi(value);
        } else if (key == "eos_token_id") {
            config.eos_token_id = std::stoi(value);
        } else if (key == "dropout_rate") {
            config.dropout_rate = std::stof(value);
        } else if (key == "layer_norm_epsilon") {
            config.layer_norm_epsilon = std::stof(value);
        }
    }
    
    return config;
}

void Config::print() const {
    std::cout << "T5 Configuration:\n";
    std::cout << "  d_model: " << d_model << "\n";
    std::cout << "  d_kv: " << d_kv << "\n";
    std::cout << "  d_ff: " << d_ff << "\n";
    std::cout << "  num_layers: " << num_layers << "\n";
    std::cout << "  num_decoder_layers: " << num_decoder_layers << "\n";
    std::cout << "  num_heads: " << num_heads << "\n";
    std::cout << "  vocab_size: " << vocab_size << "\n";
}
}