#include "model.hpp"
#include "tokenizer.hpp"
#include "weight_loader.hpp"
#include <iostream>
#include <string>

int main() {
    try {

        T5Model model;
        WeightLoader::load_weights(model, "t5_weights.bin");

        SPTokenizer tokenizer("spiece.model");

        std::string input_text = "translate English to German: Hello world. I am so happy that everything has been changed";
        std::cout << "Input: " << input_text << std::endl;

        std::vector<int> input_ids = tokenizer.encode(input_text, true);
        Tensor input({static_cast<int>(input_ids.size())});
        for (size_t i = 0; i < input_ids.size(); i++) {
            input.data[i] = static_cast<float>(input_ids[i]);
        }

        std::vector<int> output_ids = model.generate(input, 32);

        std::string output_text = tokenizer.decode(output_ids);
        std::cout << "Output: " << output_text << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
