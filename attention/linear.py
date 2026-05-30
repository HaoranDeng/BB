from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        v = self.attn_dropout(v)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).clamp_min(self.eps)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, normalizer.reciprocal())
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_dropout(self.proj(out))
