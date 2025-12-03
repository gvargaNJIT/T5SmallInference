#include "model.hpp"
#include "tokenizer.hpp"
#include "weight_loader.hpp"
#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { // Only rank 0 executes the model
        try {
            std::cout << "Starting T5 model on rank 0" << std::endl;

            T5Model model;
            WeightLoader::load_weights(model, "t5_weights.bin");

            SPTokenizer tokenizer("spiece.model");

            std::string input_text = "translate English to German: Hello world.";
            std::cout << "Input: " << input_text << std::endl;

            std::vector<int> input_ids = tokenizer.encode(input_text, true);
            Tensor input({static_cast<int>(input_ids.size())});
            for (size_t i = 0; i < input_ids.size(); i++) {
                input.data[i] = static_cast<float>(input_ids[i]);
            }

            std::vector<int> output_ids = model.generate(input, 8);

            std::string output_text = tokenizer.decode(output_ids);
            std::cout << "Output: " << output_text << std::endl;

        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Finalize();
    return 0;
}
