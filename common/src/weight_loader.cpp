#include "weight_loader.hpp"
#include <iostream>

bool WeightLoader::load_weights(T5Model &model, const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    load_tensor(file, model.encoder.embed.weight);

    for (auto &block : model.encoder.blocks)
    {
        load_block(file, block);
    }

    load_tensor(file, model.encoder.final_rms_norm.weight);

    for (auto &block : model.decoder.blocks)
    {
        load_block(file, block);
    }

    load_tensor(file, model.decoder.final_rms_norm.weight);

    model.decoder.embed.weight = model.encoder.embed.weight;
    model.lm_head.weight = model.encoder.embed.weight;

    file.close();
    return true;
}

void WeightLoader::load_tensor(std::ifstream &file, Tensor &tensor)
{
    int ndim;
    file.read(reinterpret_cast<char *>(&ndim), sizeof(int));
    tensor.shape.resize(ndim);

    file.read(reinterpret_cast<char *>(tensor.shape.data()), ndim * sizeof(int));

    int size;
    file.read(reinterpret_cast<char *>(&size), sizeof(int));
    tensor.data.resize(size);

    file.read(reinterpret_cast<char *>(tensor.data.data()), size * sizeof(float));
}

void WeightLoader::load_linear(std::ifstream &file, Linear &linear)
{
    load_tensor(file, linear.weight);
}

void WeightLoader::load_block(std::ifstream &file, T5Block &block)
{

    load_tensor(file, block.rms_norm_before_self_attention.weight);
    load_tensor(file, block.rms_norm_before_feedforward.weight);

    load_linear(file, block.self_attention_layer.q_proj);
    load_linear(file, block.self_attention_layer.k_proj);
    load_linear(file, block.self_attention_layer.v_proj);
    load_linear(file, block.self_attention_layer.o_proj);

    if (block.self_attention_layer.has_relative_bias)
    {
        load_tensor(file, block.self_attention_layer.relative_attention_bias);
    }

    if (block.is_decoder_block)
    {
        load_tensor(file, block.rms_norm_before_cross_attention.weight);

        load_linear(file, block.cross_attention_layer.q_proj);
        load_linear(file, block.cross_attention_layer.k_proj);
        load_linear(file, block.cross_attention_layer.v_proj);
        load_linear(file, block.cross_attention_layer.o_proj);
    }

    load_linear(file, block.feedforward_layer.wi);
    load_linear(file, block.feedforward_layer.wo);
}
