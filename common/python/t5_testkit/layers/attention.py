import torch
import torch.nn as nn
import numpy as np


def relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance=128):
    rp = relative_position

    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (rp > 0).long() * num_buckets
        rp = rp.abs()
    else:
        rp = torch.clamp(-rp, min=0)

    max_exact = num_buckets // 2
    is_small = rp < max_exact

    large_pos = max_exact + (
        torch.log(rp.float() / max_exact) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact)
    ).long()

    large_pos = torch.clamp(large_pos, max=num_buckets - 1)

    ret += torch.where(is_small, rp, large_pos)
    return ret


class T5Attention(nn.Module):

    def __init__(self, config, has_relative_bias=True):
        super().__init__()

        self.d_model = config["d_model"]
        self.d_kv = config["d_kv"]
        self.n_heads = config["num_heads"]
        self.inner_dim = self.n_heads * self.d_kv

        self.num_buckets = config["relative_attention_num_buckets"]
        self.bidirectional = not config.get("is_decoder", False)

        self.has_relative_bias = has_relative_bias

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    def compute_bias(self, q_len, k_len, device):
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        rel = k_pos - q_pos

        buckets = relative_position_bucket(
            rel,
            self.bidirectional,
            self.num_buckets
        ) 

        bias = self.relative_attention_bias(buckets)

        return bias.permute(2, 0, 1).unsqueeze(0)

    def forward(self, hidden_states, key_value_states=None):
        B, S, _ = hidden_states.shape

        kv = hidden_states if key_value_states is None else key_value_states
        K_len = kv.shape[1]

        q = self.q(hidden_states) 
        k = self.k(kv)
        v = self.v(kv)

        q = q.view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        k = k.view(B, K_len, self.n_heads, self.d_kv).transpose(1, 2)
        v = v.view(B, K_len, self.n_heads, self.d_kv).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))

        if self.has_relative_bias:
            pos_bias = self.compute_bias(S, K_len, hidden_states.device)
        else:
            pos_bias = torch.zeros((1, self.n_heads, S, K_len),
                                   device=hidden_states.device)

        scores = scores + pos_bias

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, S, self.inner_dim)
        out = self.o(out)

        return out, pos_bias