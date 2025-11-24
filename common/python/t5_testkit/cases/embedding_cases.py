import os
import torch
import torch.nn as nn
from utils import save_tensor_pair


class EmbeddingCases:

    def __init__(self, output_dir, config):
        self.out = os.path.join(output_dir, "embedding")
        self.vocab = config["vocab_size"]
        self.dim = config["d_model"]

    def generate(self):
        print("\nGenerating embedding tests...")
        os.makedirs(self.out, exist_ok=True)

        embed = nn.Embedding(self.vocab, self.dim)

        test_lengths = [8, 16, 32, 64, 1]

        for i, seq in enumerate(test_lengths):
            name = f"embedding_{i+1:02d}"

            idx = torch.randint(0, self.vocab, (1, seq), dtype=torch.long)

            out = embed(idx)

            save_tensor_pair(
                self.out,
                name,
                idx.float(),    
                out,
                {"weight": embed.weight}
            )

        print(f"embedding test cases generated. \n")

