import os
import struct
import numpy as np
import torch


def save_tensor_binary(file, name, tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy().astype(np.float32)

    name_bytes = name.encode("utf-8")
    file.write(struct.pack("i", len(name_bytes)))
    file.write(name_bytes)

    shape = tensor.shape
    file.write(struct.pack("i", len(shape)))
    file.write(struct.pack(f"{len(shape)}i", *shape))

    flat = tensor.flatten()
    file.write(struct.pack("i", len(flat)))
    file.write(struct.pack(f"{len(flat)}f", *flat))


def save_tensor_pair(output_dir, test_name, input_tensor, output_tensor, extra=None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{test_name}.bin")

    with open(path, "wb") as f:
        save_tensor_binary(f, "input", input_tensor)
        save_tensor_binary(f, "output", output_tensor)

        if extra:
            for name, tensor in extra.items():
                save_tensor_binary(f, name, tensor)

    print(f"Saved {test_name}")
