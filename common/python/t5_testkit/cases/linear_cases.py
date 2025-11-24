import os
import torch
import torch.nn as nn
from utils import save_tensor_pair


class LinearCases:
    def __init__(self, output_dir, config):
        self.out = os.path.join(output_dir, "linear")
        self.cfg = config

    def generate(self):
        print("\nGenerating linear tests...")
        os.makedirs(self.out, exist_ok=True)

        d_model = self.cfg["d_model"]
        d_ff = self.cfg["d_ff"]

    # Q/K/V projection (512 -> 512)
    # -------------------------------
        layer = nn.Linear(d_model, d_model, bias=False)
        x = torch.randn(1, 32, d_model)   
        out = layer(x)
        save_tensor_pair(
            self.out, "linear_01", x, out,
            {"weight": layer.weight.T}
        )

    # projection (512 -> 512)
    # -------------------------------------   
        layer = nn.Linear(d_model, d_model, bias=False)
        x = torch.randn(1, 32, d_model)
        out = layer(x)
        save_tensor_pair(
            self.out, "linear_02", x, out,
            {"weight": layer.weight.T}
        )

    # feedforward (512 -> 2048)
    # -------------------------------------   
        layer = nn.Linear(d_model, d_ff, bias=False)
        x = torch.randn(1, 32, d_model)
        out = layer(x)
        save_tensor_pair(
            self.out, "linear_03", x, out,
            {"weight": layer.weight.T}
        )

    # feedforward (2048 -> 512)
    # -------------------------------------    
        layer = nn.Linear(d_ff, d_model, bias=False)
        x = torch.randn(1, 32, d_ff)
        out = layer(x)
        save_tensor_pair(
            self.out, "linear_04", x, out,
            {"weight": layer.weight.T}
        )

        print("Linear test cases generated. \n")
