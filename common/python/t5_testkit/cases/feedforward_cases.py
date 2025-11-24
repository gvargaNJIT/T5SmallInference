import os
import torch
import torch.nn as nn
from utils import save_tensor_pair


class FeedForwardCases:
    def __init__(self, output_dir, config):
        self.out = os.path.join(output_dir, "feedforward")
        self.d_model = config["d_model"]      
        self.d_ff = config["d_ff"]       

    def generate(self):
        print("\nGenerating Feed Forward tests...")
        os.makedirs(self.out, exist_ok=True)

        wi = nn.Linear(self.d_model, self.d_ff, bias=False)
        wo = nn.Linear(self.d_ff, self.d_model, bias=False)

        seq_lengths = [32, 64, 8, 16, 1]

        for i, seq in enumerate(seq_lengths):
            name = f"feedforward_{i+1:02d}"

            x = torch.randn(1, seq, self.d_model)

            h = torch.relu(wi(x))
            y = wo(h)

            save_tensor_pair(
                self.out,
                name,
                x,
                y,
                {
                    "wi_weight": wi.weight.T,
                    "wo_weight": wo.weight.T
                }
            )


        print(f"feedforward test cases generated. \n")

