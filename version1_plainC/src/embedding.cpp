#include "embedding.hpp"

Embedding::Embedding(int num_emb, int emb_dim)
    : num_embeddings(num_emb), embedding_dim(emb_dim) {
    weight = Tensor::randn({num_emb, emb_dim}, 0.0f, 0.02f);
}

Tensor Embedding::forward(const Tensor& indices) {
    int batch = indices.shape[0];
    int seq_len = indices.shape[1];
    Tensor result({batch, seq_len, embedding_dim});
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int idx = static_cast<int>(indices.data[b * seq_len + s]);
            for (int d = 0; d < embedding_dim; d++) {
                result.data[b * seq_len * embedding_dim +
                            s * embedding_dim + d] =
                    weight.data[idx * embedding_dim + d];
            }
        }
    }

    return result;
}
