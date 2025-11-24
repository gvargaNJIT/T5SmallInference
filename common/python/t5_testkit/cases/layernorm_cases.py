import os
import torch
from utils import save_tensor_pair


class LayerNormCases:
    def __init__(self, output_dir, config):
        self.out = os.path.join(output_dir, "layer_norm")
        self.eps = config["layer_norm_epsilon"]
        self.hidden = config["d_model"]

    def rmsnorm(self, x, weight):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * weight

    def generate(self):
        print("\nGenerating RMSNorm tests...")
        os.makedirs(self.out, exist_ok=True)

        H = self.hidden

        seq_lengths = [32, 64, 8, 16, 1]


        for i in range(5):
            seq = seq_lengths[i]
            w = torch.randn(H) * 0.05 + 1.0

            x = torch.randn(1, seq, H)
            y = self.rmsnorm(x, w)
            name = f"rmsnorm_{i+1:02d}"

            save_tensor_pair(
                self.out,
                name,
                x,
                y,
                {"weight": w}
            )

        print(f"layer norm test cases generated. \n")

