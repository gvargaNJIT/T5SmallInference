import os
import torch
from utils import save_tensor_pair, save_tensor_binary


class TensorCases:
    def __init__(self, output_dir):
        self.out = os.path.join(output_dir, "tensor")

    def generate(self):
        print("\nGenerating Tensor tests...")
        os.makedirs(self.out, exist_ok=True)

    
    # permute
    # -------------------------------------
        permute_cases = [
            (torch.randn(2, 3, 4), (1, 0, 2)),
            (torch.randn(1, 4, 5, 6), (3, 2, 1, 0)),
            (torch.randn(3, 2), (1, 0)),
            (torch.randn(2, 2, 2, 2), (2, 3, 1, 0)),
            (torch.randn(5, 3, 2), (2, 1, 0)),
        ]

        for i, (x, perm) in enumerate(permute_cases, 1):
            save_tensor_pair(self.out, f"permute_{i}", x, x.permute(*perm))

    # softmax
    # -------------------------------------
        softmax_shapes = [
            (2, 3, 10),
            (1, 5, 8),
            (4, 2, 6),
            (3, 3, 3),
            (2, 10, 5),
        ]

        for i, (a, b, c) in enumerate(softmax_shapes, 1):
            x = torch.randn(a, b, c)
            save_tensor_pair(self.out, f"softmax_{i}", x, torch.softmax(x, dim=-1))

    # matmul
    # -------------------------------------
        matmul_shapes = [
            (3, 4, 4, 5),
            (2, 6, 6, 3),
            (1, 10, 10, 8),
            (5, 3, 3, 7),
            (4, 2, 2, 6),
        ]

        for i, (a0, a1, b0, b1) in enumerate(matmul_shapes, 1):
            a = torch.randn(a0, a1)
            b = torch.randn(b0, b1)
            out = torch.matmul(a, b)
            fname = f"matmul_{i}.bin"

            with open(os.path.join(self.out, fname), "wb") as f:
                save_tensor_binary(f, "a", a)
                save_tensor_binary(f, "b", b)
                save_tensor_binary(f, "output", out)


    # operator+
    # -------------------------------------
        add_shapes = [
            (2, 3, 4),
            (5, 2),
            (1, 10),
            (3, 3, 3),
            (4, 4),
        ]

        for i, shape in enumerate(add_shapes, 1):
            a = torch.randn(*shape)
            b = torch.randn(*shape)
            fname = f"add_{i}.bin"
            with open(os.path.join(self.out, fname), "wb") as f:
                save_tensor_binary(f, "a", a)
                save_tensor_binary(f, "b", b)
                save_tensor_binary(f, "output", a + b)
        
        
        print(f"Tensor test cases generated. \n")

        