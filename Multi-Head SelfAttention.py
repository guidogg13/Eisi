import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # maschera causale [T, T]
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, time, channels

        qkv = self.qkv(x)              # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1) # ognuno (B, T, C)

        # reshape per teste
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            # (B, n_heads, T, head_dim)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # attenzione scaled dot‑product
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        y = self.proj(y)
        y = self.dropout(y)
        return y
