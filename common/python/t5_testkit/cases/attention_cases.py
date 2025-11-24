import os
import torch

from layers.attention import T5Attention
from utils import save_tensor_pair


class AttentionCases:

    def __init__(self, output_dir, config):
        self.out = os.path.join(output_dir, "attention")
        self.cfg = config

        os.makedirs(self.out, exist_ok=True)

        self.attn = T5Attention(self.cfg, has_relative_bias=True)

    def _weights(self, pos_bias):
        return {
            "q_weight": self.attn.q.weight.T,
            "k_weight": self.attn.k.weight.T,
            "v_weight": self.attn.v.weight.T,
            "o_weight": self.attn.o.weight.T,
            "rel_bias_weight": self.attn.relative_attention_bias.weight,
            "pos_bias": pos_bias,
        }
    

    def generate(self):
        print("\nGenerating Attention tests...")

    #self attention
    # -------------------------------------
        for idx in range(1, 6):
            B = 1
            S = 512
            d = self.cfg["d_model"]

            x = torch.randn(B, S, d)
            out, pb = self.attn(x)

            name = f"self_attention_{idx:02d}"
            save_tensor_pair(self.out, name, x, out, extra=self._weights(pb))



    #cross attention
    # -------------------------------------
        for idx in range(1, 6):
            B = 1
            S_dec = 512
            S_enc = 512
            d = self.cfg["d_model"]

            decoder = torch.randn(B, S_dec, d)
            encoder = torch.randn(B, S_enc, d)

            out, pb = self.attn(decoder, key_value_states=encoder)

            name = f"cross_attention_{idx:02d}"

            extra = self._weights(pb)
            extra["encoder_input"] = encoder

            save_tensor_pair(self.out, name, decoder, out, extra=extra)

        


        print(f"Attention test cases generated. \n")

   
   
    