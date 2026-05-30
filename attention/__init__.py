from __future__ import annotations

from torch import nn

from attention.linear import LinearAttention
from attention.monarch import MonarchAttention
from attention.standard import StandardAttention

ATTENTION_REGISTRY = {
    "standard": StandardAttention,
    "linear": LinearAttention,
    "monarch": MonarchAttention,
}


def build_attention(
    name: str,
    dim: int,
    num_heads: int,
    attn_dropout: float = 0.0,
    proj_dropout: float = 0.0,
) -> nn.Module:
    key = name.lower()
    if key not in ATTENTION_REGISTRY:
        available = ", ".join(sorted(ATTENTION_REGISTRY))
        raise ValueError(f"Unknown attention {name!r}; available: {available}")
    return ATTENTION_REGISTRY[key](
        dim=dim,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
    )


__all__ = [
    "LinearAttention",
    "MonarchAttention",
    "StandardAttention",
    "build_attention",
]
