#include "weight_loader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <algorithm>
#include <cctype>

namespace t5 {

std::vector<size_t> WeightLoader::parse_shape_file(const std::string& shape_file) {
    std::ifstream file(shape_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open shape file: " << shape_file << std::endl;
        return {};
    }
    
    std::vector<size_t> shape;
    std::string line;
    std::getline(file, line);
    line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
    line.erase(remove(line.begin(), line.end(), '('), line.end());
    line.erase(remove(line.begin(), line.end(), ')'), line.end());
    line.erase(remove(line.begin(), line.end(), '['), line.end());
    line.erase(remove(line.begin(), line.end(), ']'), line.end());
    
    if (line.empty()) {
        std::cerr << "Error: Empty shape file: " << shape_file << std::endl;
        return {};
    }
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
        value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
        if (!value.empty() && std::isdigit(value[0])) {
            try {
                shape.push_back(std::stoul(value));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing shape value '" << value 
                          << "' in file " << shape_file << std::endl;
                return {};
            }
        }
    }
    
    return shape;
}

Tensor WeightLoader::load_single_weight(const std::string& bin_file,const std::string& shape_file) {
    auto shape = parse_shape_file(shape_file);
    if (shape.empty()) {
        std::cerr << "Error: Invalid shape for " << bin_file << std::endl;
        return Tensor();
    }
    Tensor tensor(shape);
    std::ifstream file(bin_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << bin_file << std::endl;
        return Tensor();
    }

    file.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
    return tensor;
}

std::unordered_map<std::string, Tensor> WeightLoader::load_weights(const std::string& weights_dir) {
    std::unordered_map<std::string, Tensor> weights;
    std::cout << "Loading weights from: " << weights_dir << std::endl;
    DIR* dir = opendir(weights_dir.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory: " << weights_dir << std::endl;
        return weights;
    }
    struct dirent* entry;
    std::vector<std::string> bin_files;
    
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.size() > 4 && 
            filename.substr(filename.size() - 4) == ".bin" &&
            filename.find(".bin.shape") == std::string::npos) {
            bin_files.push_back(filename);
        }
    }
    closedir(dir);
    std::cout << "Found " << bin_files.size() << " weight files" << std::endl;
    int loaded = 0;
    for (const auto& bin_filename : bin_files) {
        std::string weight_name = bin_filename.substr(0, bin_filename.size() - 4);
        std::string converted_name;
        for (size_t i = 0; i < weight_name.size(); i++) {
            if (weight_name[i] == '_') {
                converted_name += '.';
            } else {
                converted_name += weight_name[i];
            }
        }

        std::string bin_path = weights_dir + "/" + bin_filename;
        std::string shape_path = bin_path + ".shape";

        Tensor tensor = load_single_weight(bin_path, shape_path);
        
        if (tensor.size() > 0) {
            weights[converted_name] = std::move(tensor);
            loaded++;
            
            if (loaded <= 5 || loaded % 20 == 0) {
                std::cout << "  Loaded [" << loaded << "/" << bin_files.size() << "]: " 
                          << converted_name << " ";
                tensor.print_info();
            }
        }
    }
    
    std::cout << "\nSuccessfully loaded " << weights.size() << " tensors" << std::endl;
    
    return weights;
}

void WeightLoader::print_weight_info(
    const std::unordered_map<std::string, Tensor>& weights
) {
    std::cout << "\n=== Weight Statistics ===" << std::endl;
    std::cout << "Total tensors: " << weights.size() << std::endl;
    
    size_t total_params = 0;
    for (const auto& pair : weights) {
        total_params += pair.second.size();
    }
    
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Memory size: " << (total_params * sizeof(float)) / (1024 * 1024) << " MB" << std::endl;

    std::cout << "\nSample weights:" << std::endl;
    int count = 0;
    for (const auto& pair : weights) {
        if (count++ >= 5) break;
        std::cout << "  " << pair.first << ": ";
        pair.second.print_info();
    }
}

}