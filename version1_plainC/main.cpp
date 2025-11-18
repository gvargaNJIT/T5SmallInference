#include "config.hpp"
#include "serial_model.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "T5-Small Inference - VERSION 1: Single CPU\n\n";
    auto start_total = std::chrono::high_resolution_clock::now();
    try {
        std::cout << "Loading configuration...\n";
        t5::Config config = t5::Config::load("weights/config.json");
        config.print();
        std::cout << std::endl;
        std::cout << "Creating model...\n";
        t5::serial::T5Model model(config);
        std::cout << "Model created\n\n";
        std::cout << "Loading pre-trained weights...\n";
        auto start_load = std::chrono::high_resolution_clock::now();
        model.load_weights("weights/processed_weights");
        auto end_load = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration<double>(end_load - start_load).count();
        std::cout << "Weights loaded in " << load_time << " seconds\n\n";
        std::cout << "Loading tokenizer...\n";
        t5::Tokenizer tokenizer;
        if (!tokenizer.load_vocab("weights/vocab.json")) {
            std::cerr << "Failed to load tokenizer\n";
            return 1;
        }
        std::cout << "Tokenizer loaded\n\n";
        std::string input_text = "translate English to German: The house is wonderful.";
        std::cout << "Input: " << input_text << "\n\n";
        std::vector<int> input_ids = tokenizer.encode(input_text);
        std::cout << "Encoded to " << input_ids.size() << " tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), input_ids.size()); i++) {
            std::cout << input_ids[i] << " ";
        }

        if (input_ids.size() > 10) {std::cout << "...";}
        std::cout << "\n\n";
        std::cout << "Running inference (greedy decoding)...\n";
        std::cout << "This will take a while on CPU...\n\n";
        auto start_infer = std::chrono::high_resolution_clock::now();
        std::vector<int> output_ids = model.generate(input_ids, 50);
        auto end_infer = std::chrono::high_resolution_clock::now();
        auto infer_time = std::chrono::duration<double>(end_infer - start_infer).count();
        std::string output_text = tokenizer.decode(output_ids);
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double>(end_total - start_total).count();

        std::cout << "RESULTS\n";
        std::cout << "--------------------------------------\n";
        std::cout << "Input:  " << input_text << "\n";
        std::cout << "Output: " << output_text << "\n";
        std::cout << "\n--------------------------------------\n";
        std::cout << "TIMING (Version 1: Serial CPU)\n";
        std::cout << "--------------------------------------\n";
        std::cout << "Weight loading:  " << load_time << " seconds\n";
        std::cout << "Inference time:  " << infer_time << " seconds\n";
        std::cout << "Total time:      " << total_time << " seconds\n";
        std::cout << "Tokens/second:   " << output_ids.size() / infer_time << "\n";
        std::cout << "Tokens generated: " << output_ids.size() << "\n";
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}