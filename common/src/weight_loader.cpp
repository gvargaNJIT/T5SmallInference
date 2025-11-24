#include "weight_loader.hpp"

bool WeightLoader::load_weights(T5Model &model, const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }
    load_tensor(file, model.encoder.embed.weight);
    for (auto &block : model.encoder.blocks) {
        load_block(file, block);
    }
    load_tensor(file, model.encoder.final_layer_norm.weight);
    for (auto &block : model.decoder.blocks) {
        load_block(file, block);
    }
    load_tensor(file, model.decoder.final_layer_norm.weight);
    model.decoder.embed.weight = model.encoder.embed.weight;
    model.lm_head.weight       = model.encoder.embed.weight;
    file.close();
    return true;
}

void WeightLoader::load_tensor(std::ifstream &file, Tensor &tensor) {
    int ndim;
    file.read(reinterpret_cast<char *>(&ndim), sizeof(int));
    std::vector<int> shape(ndim);
    file.read(reinterpret_cast<char *>(shape.data()), ndim * sizeof(int));
    int size;
    file.read(reinterpret_cast<char *>(&size), sizeof(int));
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    tensor.shape = std::move(shape);
    tensor.data  = std::move(data);
}

void WeightLoader::load_linear(std::ifstream &file, Linear &linear) {
    load_tensor(file, linear.weight);
    if (linear.use_bias) {
        load_tensor(file, linear.bias);
    }
}

void WeightLoader::load_block(std::ifstream &file, T5Block &block) {
    load_tensor(file, block.layer_norm_self_attn.weight);
    load_tensor(file, block.layer_norm_ff.weight);
    load_linear(file, block.self_attn.q_proj);
    load_linear(file, block.self_attn.k_proj);
    load_linear(file, block.self_attn.v_proj);
    load_linear(file, block.self_attn.o_proj);
    if (block.self_attn.has_relative_bias) {
        load_tensor(file, block.self_attn.relative_attention_bias);
    }

    if (block.is_decoder && block.cross_attn) {
        load_tensor(file, block.layer_norm_cross_attn->weight);
        load_linear(file, block.cross_attn->q_proj);
        load_linear(file, block.cross_attn->k_proj);
        load_linear(file, block.cross_attn->v_proj);
        load_linear(file, block.cross_attn->o_proj);
    }

    load_linear(file, block.ff.wi);
    load_linear(file, block.ff.wo);
}
