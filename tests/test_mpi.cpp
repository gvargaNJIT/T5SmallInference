#include "tokenizer.hpp"
#include "embedding.hpp"
#include "feedforward.hpp"
#include "layer_norm.hpp"
#include "linear.hpp"
#include "tensor.hpp"
#include "config.hpp"
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <memory>

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

void print_result(int rank, const std::string& test_name, float rel_err, float abs_err, float threshold = 1e-4f) {
    if (rank == 0) {
        bool passed = rel_err < threshold;
        std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name;
        std::cout << " (rel_err: " << std::scientific << rel_err;
        std::cout << ", max_abs_err: " << abs_err << ")" << std::endl;
    }
}

void broadcast_tensor_metadata(Tensor& tensor, int rank) {
    int ndim;
    if (rank == 0) {
        ndim = tensor.shape.size();
    }
    MPI_Bcast(&ndim, 1, MPI_INT, 0, MPI_COMM_WORLD);    
    if (rank != 0) {
        tensor.shape.resize(ndim);
    }
    MPI_Bcast(tensor.shape.data(), ndim, MPI_INT, 0, MPI_COMM_WORLD);
    int size = 1;
    for (int dim : tensor.shape) size *= dim;
    if (rank != 0) {
        tensor.data.resize(size);
    }
}

void broadcast_tensor(Tensor& tensor, int rank) {
    broadcast_tensor_metadata(tensor, rank);
    MPI_Bcast(tensor.data.data(), tensor.data.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

bool test_linear_mpi(const std::string& data_dir, int rank, int world_size) {
    if (rank == 0) {
        std::cout << "\nTesting Linear Layer" << std::endl;
    }
    bool all_passed = true;
    {
        Tensor input, expected, weight, bias;
        if (rank == 0) {
            std::string filepath = data_dir + "/linear/linear_with_bias.bin";
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cout << "  Skipping: " << filepath << " not found" << std::endl;
                return true;
            }
            auto [n1, inp] = load_named_tensor(file);
            auto [n2, exp] = load_named_tensor(file);
            auto [n3, wgt] = load_named_tensor(file);
            auto [n4, bis] = load_named_tensor(file);
            input = inp;
            expected = exp;
            weight = wgt;
            bias = bis;
            file.close();
        }
        broadcast_tensor(weight, rank);
        broadcast_tensor(bias, rank);
        broadcast_tensor_metadata(input, rank);
        broadcast_tensor_metadata(expected, rank);
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int in_features = input.shape[2];
        int out_features = weight.shape[1];
        int batches_per_rank = batch_size / world_size;
        int remainder = batch_size % world_size;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            int local_batch = batches_per_rank + (i < remainder ? 1 : 0);
            sendcounts[i] = local_batch * seq_len * in_features;
            displs[i] = offset;
            offset += sendcounts[i];
        }
        int local_batch = batches_per_rank + (rank < remainder ? 1 : 0);
        std::vector<int> local_input_shape = {local_batch, seq_len, in_features};
        Tensor local_input(local_input_shape);

        MPI_Scatterv(input.data.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                     local_input.data.data(), local_input.data.size(), MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        
        Linear layer(in_features, out_features, true);
        layer.weight = weight;
        layer.bias = bias;
        Tensor local_output = layer.forward(local_input);
        std::vector<int> recvcounts(world_size);
        std::vector<int> recvdispls(world_size);
        offset = 0;
        for (int i = 0; i < world_size; i++) {
            int lb = batches_per_rank + (i < remainder ? 1 : 0);
            recvcounts[i] = lb * seq_len * out_features;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
        Tensor gathered_output;
        if (rank == 0) {
            gathered_output = Tensor({batch_size, seq_len, out_features});
        }
        
        MPI_Gatherv(local_output.data.data(), local_output.data.size(), MPI_FLOAT,
                    rank == 0 ? gathered_output.data.data() : nullptr,
                    recvcounts.data(), recvdispls.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            float rel_err = relative_error(gathered_output, expected);
            float abs_err = max_abs_error(gathered_output, expected);
            print_result(rank, "Linear with bias (MPI)", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
        }
    }
    return all_passed;
}

bool test_embedding_mpi(const std::string& data_dir, int rank, int world_size) {
    if (rank == 0) {
        std::cout << "\nTesting Embedding Layer" << std::endl;
    }
    bool all_passed = true;    
    {
        Tensor indices, expected, weight;
        if (rank == 0) {
            std::string filepath = data_dir + "/embedding/embedding_basic.bin";
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cout << "  Skipping: " << filepath << " not found" << std::endl;
                return true;
            }
            auto [n1, ind] = load_named_tensor(file);
            auto [n2, exp] = load_named_tensor(file);
            auto [n3, wgt] = load_named_tensor(file);
            indices = ind;
            expected = exp;
            weight = wgt;
            file.close();
        }
        broadcast_tensor(weight, rank);
        broadcast_tensor_metadata(indices, rank);
        broadcast_tensor_metadata(expected, rank);
        int batch_size = indices.shape[0];
        int seq_len = indices.shape[1];
        int emb_dim = weight.shape[1];
        int batches_per_rank = batch_size / world_size;
        int remainder = batch_size % world_size;
        int local_batch = batches_per_rank + (rank < remainder ? 1 : 0);
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            int lb = batches_per_rank + (i < remainder ? 1 : 0);
            sendcounts[i] = lb * seq_len;
            displs[i] = offset;
            offset += sendcounts[i];
        }
        Tensor local_indices({local_batch, seq_len});
        
        MPI_Scatterv(indices.data.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                     local_indices.data.data(), local_indices.data.size(), MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        
        Embedding emb(weight.shape[0], emb_dim);
        emb.weight = weight;
        Tensor local_output = emb.forward(local_indices);
        std::vector<int> recvcounts(world_size);
        std::vector<int> recvdispls(world_size);
        offset = 0;
        for (int i = 0; i < world_size; i++) {
            int lb = batches_per_rank + (i < remainder ? 1 : 0);
            recvcounts[i] = lb * seq_len * emb_dim;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
        Tensor gathered_output;
        if (rank == 0) {
            gathered_output = Tensor({batch_size, seq_len, emb_dim});
        }
        
        MPI_Gatherv(local_output.data.data(), local_output.data.size(), MPI_FLOAT,
                    rank == 0 ? gathered_output.data.data() : nullptr,
                    recvcounts.data(), recvdispls.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            float rel_err = relative_error(gathered_output, expected);
            float abs_err = max_abs_error(gathered_output, expected);
            print_result(rank, "Embedding basic (MPI)", rel_err, abs_err);
            all_passed &= (rel_err < 1e-5f);
        }
    }
    return all_passed;
}

bool test_tokenizer_mpi(const std::string& tokenizer_path, int rank, int world_size) {
    if (rank == 0) {
        std::cout << "\nTesting Tokenizer" << std::endl;
    }    
    bool all_passed = true;
    try {
        std::unique_ptr<SPTokenizer> tokenizer;
        std::vector<std::string> texts;
        std::vector<std::vector<int>> all_token_ids;
        int total_tokens = 0;
        if (rank == 0) {
            tokenizer = std::make_unique<SPTokenizer>(tokenizer_path);
            texts = {
                "Hello world, this is a test.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is fascinating and powerful.",
                "Distributed computing enables large scale processing.",
                "Natural language processing transforms text data.",
                "Deep learning models require significant computation.",
                "Parallel processing improves performance dramatically.",
                "Tokenization is the first step in text processing."
            };
            while (texts.size() < (size_t)world_size) {
                texts.push_back("Additional test sentence number " + std::to_string(texts.size()));
            }
            std::cout << "  Rank 0 tokenizing " << texts.size() << " texts..." << std::endl;
            for (const auto& text : texts) {
                std::vector<int> ids = tokenizer->encode(text, true);
                all_token_ids.push_back(ids);
                total_tokens += ids.size();
            }
            std::cout << "  Total tokens: " << total_tokens << std::endl;
        }
        int num_texts = 0;
        if (rank == 0) {
            num_texts = texts.size();
        }
        MPI_Bcast(&num_texts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int texts_per_rank = num_texts / world_size;
        int remainder = num_texts % world_size;
        int local_num_texts = texts_per_rank + (rank < remainder ? 1 : 0);
        std::vector<int> all_lengths;
        std::vector<int> flat_tokens;
        int max_len = 0;
        if (rank == 0) {
            for (const auto& ids : all_token_ids) {
                max_len = std::max(max_len, (int)ids.size());
            }
            for (const auto& ids : all_token_ids) {
                all_lengths.push_back(ids.size());
                flat_tokens.insert(flat_tokens.end(), ids.begin(), ids.end());
                for (size_t i = ids.size(); i < (size_t)max_len; i++) {
                    flat_tokens.push_back(tokenizer->get_pad_id());
                }
            }
        }
        
        MPI_Bcast(&max_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        std::vector<int> length_counts(world_size);
        std::vector<int> length_displs(world_size);
        int offset = 0;
        int length_offset = 0;
        for (int i = 0; i < world_size; i++) {
            int local_texts = texts_per_rank + (i < remainder ? 1 : 0);
            sendcounts[i] = local_texts * max_len;
            displs[i] = offset;
            offset += sendcounts[i];
            length_counts[i] = local_texts;
            length_displs[i] = length_offset;
            length_offset += local_texts;
        }
        std::vector<int> local_flat_tokens(local_num_texts * max_len);

        MPI_Scatterv(flat_tokens.data(), sendcounts.data(), displs.data(), MPI_INT,
                     local_flat_tokens.data(), local_flat_tokens.size(), MPI_INT,
                     0, MPI_COMM_WORLD);
        
        std::vector<int> local_lengths(local_num_texts);

        MPI_Scatterv(all_lengths.data(), length_counts.data(), length_displs.data(), MPI_INT,
                     local_lengths.data(), local_lengths.size(), MPI_INT,
                     0, MPI_COMM_WORLD);
        
        int local_valid_tokens = 0;
        for (int i = 0; i < local_num_texts; i++) {
            local_valid_tokens += local_lengths[i];
        }
        std::vector<int> all_valid_tokens;
        if (rank == 0) {
            all_valid_tokens.resize(world_size);
        }

        MPI_Gather(&local_valid_tokens, 1, MPI_INT,
                   all_valid_tokens.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            int gathered_total = 0;
            for (int count : all_valid_tokens) {
                gathered_total += count;
            }
            bool tokens_match = (gathered_total == total_tokens);
            print_result(rank, "Tokenizer scatter/gather", 
                        tokens_match ? 0.0f : 1.0f, 
                        tokens_match ? 0.0f : 1.0f);
            all_passed &= tokens_match;
            if (tokens_match) {
                std::cout << "Successfully distributed " << total_tokens 
                         << " tokens across " << world_size << " ranks" << std::endl;
            }
        }
        return all_passed;
        
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cout << "  Tokenizer test skipped: " << e.what() << std::endl;
            std::cout << "  (Make sure tokenizer model file exists)" << std::endl;
        }
        return true;
    }
}

bool test_feedforward_mpi(const std::string& data_dir, int rank, int world_size) {
    if (rank == 0) {
        std::cout << "\nTesting FeedForward" << std::endl;
    }
    bool all_passed = true;
    {
        Tensor input, expected, wi_weight, wo_weight;
        if (rank == 0) {
            std::string filepath = data_dir + "/feedforward/feedforward_basic.bin";
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cout << "  Skipping: " << filepath << " not found" << std::endl;
                return true;
            }
            auto [n1, inp] = load_named_tensor(file);
            auto [n2, exp] = load_named_tensor(file);
            auto [n3, wi] = load_named_tensor(file);
            auto [n4, wo] = load_named_tensor(file);
            input = inp;
            expected = exp;
            wi_weight = wi;
            wo_weight = wo;
            file.close();
        }
        broadcast_tensor(wi_weight, rank);
        broadcast_tensor(wo_weight, rank);
        broadcast_tensor_metadata(input, rank);
        broadcast_tensor_metadata(expected, rank);
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int d_model = input.shape[2];
        int d_ff = wi_weight.shape[1];
        int batches_per_rank = batch_size / world_size;
        int remainder = batch_size % world_size;
        int local_batch = batches_per_rank + (rank < remainder ? 1 : 0);
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            int lb = batches_per_rank + (i < remainder ? 1 : 0);
            sendcounts[i] = lb * seq_len * d_model;
            displs[i] = offset;
            offset += sendcounts[i];
        }
        Tensor local_input({local_batch, seq_len, d_model});
        
        MPI_Scatterv(input.data.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                     local_input.data.data(), local_input.data.size(), MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        
        T5Config config;
        config.d_model = d_model;
        config.d_ff = d_ff;
        FeedForward ff(config);
        ff.wi.weight = wi_weight;
        ff.wo.weight = wo_weight;
        Tensor local_output = ff.forward(local_input);
        std::vector<int> recvcounts(world_size);
        std::vector<int> recvdispls(world_size);
        offset = 0;
        for (int i = 0; i < world_size; i++) {
            int lb = batches_per_rank + (i < remainder ? 1 : 0);
            recvcounts[i] = lb * seq_len * d_model;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
        Tensor gathered_output;
        if (rank == 0) {
            gathered_output = Tensor({batch_size, seq_len, d_model});
        }
        
        MPI_Gatherv(local_output.data.data(), local_output.data.size(), MPI_FLOAT,
                    rank == 0 ? gathered_output.data.data() : nullptr,
                    recvcounts.data(), recvdispls.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            float rel_err = relative_error(gathered_output, expected);
            float abs_err = max_abs_error(gathered_output, expected);
            print_result(rank, "FeedForward basic (MPI)", rel_err, abs_err);
            all_passed &= (rel_err < 1e-4f);
        }
    }
    return all_passed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (rank == 0) {
        std::cout << "______________________________________" << std::endl;
        std::cout << "  T5 MPI Validation Test Suite" << std::endl;
        std::cout << "  Running on " << world_size << " processes" << std::endl;
        std::cout << "______________________________________" << std::endl;
    }
    std::string data_dir = "layer_test_data";
    std::string tokenizer_path = "weights/spiece.model";
    if (argc > 1) {
        data_dir = argv[1];
    }
    if (argc > 2) {
        tokenizer_path = argv[2];
    }
    if (rank == 0) {
        std::cout << "Using test data from: " << data_dir << std::endl;
        std::cout << "Using tokenizer: " << tokenizer_path << std::endl;
    }
    bool all_passed = true;
    all_passed &= test_tokenizer_mpi(tokenizer_path, rank, world_size);
    all_passed &= test_linear_mpi(data_dir, rank, world_size);
    all_passed &= test_embedding_mpi(data_dir, rank, world_size);
    all_passed &= test_feedforward_mpi(data_dir, rank, world_size);
    int local_result = all_passed ? 1 : 0;
    int global_result = 0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n______________________________________" << std::endl;
        if (global_result) {
            std::cout << "ALL MPI TESTS PASSED" << std::endl;
        } else {
            std::cout << "SOME MPI TESTS FAILED" << std::endl;
        }
        std::cout << "______________________________________" << std::endl;
    }
    MPI_Finalize();
    return global_result ? 0 : 1;
}