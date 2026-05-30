from __future__ import annotations

import math

import torch
from torch import nn

from attention.standard import StandardAttention


class MonarchAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        block_size: int | None = None,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.local_attn = StandardAttention(dim, num_heads, attn_dropout, proj_dropout)
        self.global_attn = StandardAttention(dim, num_heads, attn_dropout, proj_dropout)
        self.mix = nn.Linear(dim * 2, dim)

    def infer_block_size(self, tokens: int) -> int:
        if self.block_size is not None:
            return min(self.block_size, tokens)
        root = int(math.sqrt(tokens))
        for candidate in range(root, 0, -1):
            if tokens % candidate == 0:
                return candidate
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        block = self.infer_block_size(tokens)
        if block <= 1 or block >= tokens or tokens % block != 0:
            return self.local_attn(x)

        groups = tokens // block
        local = x.reshape(batch * groups, block, channels)
        local = self.local_attn(local).reshape(batch, groups, block, channels)

        global_tokens = local.transpose(1, 2).reshape(batch * block, groups, channels)
        global_tokens = self.global_attn(global_tokens)
        global_tokens = global_tokens.reshape(batch, block, groups, channels).transpose(1, 2)
        out = torch.cat([local, global_tokens], dim=-1)
        out = self.mix(out)
        return out.reshape(batch, tokens, channels)
