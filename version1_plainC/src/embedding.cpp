#include "embedding.hpp"

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) 
{
    weight = Tensor({num_emb, emb_dim}, 0.f);
}

Tensor Embedding::forward(const Tensor& indices) 
{
    int seq_len = indices.shape[0];

    Tensor result({seq_len, embedding_dim});

    for (int s = 0; s < seq_len; s++) {
        int idx = static_cast<int>(indices.data[s]);

        for (int d = 0; d < embedding_dim; d++) {
            result.data[s * embedding_dim + d] =
                weight.data[idx * embedding_dim + d];
        }
    }

    return result;
}
