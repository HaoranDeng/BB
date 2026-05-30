from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StandardAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = attn_dropout
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_dropout(self.proj(out))
