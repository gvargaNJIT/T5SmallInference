#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "tensor.hpp"

class Embedding {
public:
    Tensor weight;
    int num_embeddings;
    int embedding_dim;
    Embedding(int num_emb, int emb_dim);
    Tensor forward(const Tensor& indices);
};

#endif
