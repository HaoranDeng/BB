from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t * 1000, self.frequency_embedding_size))


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(context_dim, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(batch, context.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        context_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, context_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_ratio)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(
            6, dim=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


@dataclass
class ImageGenDiTConfig:
    latent_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    out_channels: int = 4
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    context_dim: int = 768
    gradient_checkpointing: bool = True


class ImageGenDiT(nn.Module):
    def __init__(
        self,
        latent_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        context_dim: int = 768,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError("latent_size must be divisible by patch_size")
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = latent_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.time_embed = TimestepEmbedder(hidden_size)
        self.text_pool = nn.Linear(context_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    context_dim=context_dim,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.out_channels, -1))
        nn.init.zeros_(self.patch_embed.bias)

        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.ada_ln[-1].weight)
        nn.init.zeros_(self.final_layer.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(1)
        x = self.patch_embed(latents).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cond = self.time_embed(timesteps) + self.text_pool(text_embeds.mean(dim=1))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond, text_embeds, use_reentrant=False)
            else:
                x = block(x, cond, text_embeds)

        x = self.final_layer(x, cond)
        return self.unpatchify(x)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patch = self.patch_size
        grid = self.grid_size
        channels = self.out_channels
        x = x.reshape(batch, grid, grid, patch, patch, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(batch, channels, grid * patch, grid * patch)
