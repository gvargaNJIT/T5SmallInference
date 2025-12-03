#include "model.hpp"
#include "tokenizer.hpp"
#include "weight_loader.hpp"
#include <iostream>
#include <string>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    try {
        T5Model model;
        
        WeightLoader::load_weights(model, "t5_weights.bin");

        int input_size = 0;
        std::vector<int> input_ids;
        
        if (rank == 0) {
            SPTokenizer tokenizer("spiece.model");
            std::string input_text = "translate English to German: Hello world.";
            std::cout << "Input: " << input_text << std::endl;
            input_ids = tokenizer.encode(input_text, true);
            input_size = input_ids.size();
        }
        
        MPI_Bcast(&input_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            input_ids.resize(input_size, 0);
        }
        
        MPI_Bcast(input_ids.data(), input_size, MPI_INT, 0, MPI_COMM_WORLD);
        
        Tensor input({input_size});
        for (int i = 0; i < input_size; i++) {
            input.data[i] = static_cast<float>(input_ids[i]);
        }

        std::vector<int> output_ids = model.generate(input, 8);

        if (rank == 0) {
            SPTokenizer tokenizer("spiece.model");
            std::string output_text = tokenizer.decode(output_ids);
            std::cout << "Output: " << output_text << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << " Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
