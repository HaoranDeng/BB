from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from attention import build_attention


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

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(timesteps, self.frequency_embedding_size)
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(self, labels: torch.Tensor) -> torch.Tensor:
        if self.dropout_prob <= 0:
            return labels
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels: torch.Tensor, train: bool) -> torch.Tensor:
        if train:
            labels = self.token_drop(labels)
        return self.embedding(labels)


class MLP(nn.Module):
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


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attention: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = build_attention(attention, hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(
            6, dim=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class ClassConditionalDiT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False,
        attention: str = "standard",
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.grid_size = img_size // patch_size
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
        self.label_embed = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, attention) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
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
        nn.init.normal_(self.label_embed.embedding.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.ada_ln[-1].weight)
        nn.init.zeros_(self.final_layer.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cond = self.time_embed(timesteps) + self.label_embed(labels, train=self.training)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond, use_reentrant=False)
            else:
                x = block(x, cond)

        x = self.final_layer(x, cond)
        return self.unpatchify(x)

    def forward_with_cfg(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale == 1.0:
            return self.forward(images, timesteps, labels)
        combined = torch.cat([images, images], dim=0)
        t_combined = torch.cat([timesteps, timesteps], dim=0)
        labels_uncond = torch.full_like(labels, self.num_classes)
        labels_combined = torch.cat([labels, labels_uncond], dim=0)
        model_out = self.forward(combined, t_combined, labels_combined)
        cond_eps, uncond_eps = torch.split(model_out, images.shape[0], dim=0)
        return uncond_eps + cfg_scale * (cond_eps - uncond_eps)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patch = self.patch_size
        grid = self.grid_size
        channels = self.in_channels
        x = x.reshape(batch, grid, grid, patch, patch, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(batch, channels, grid * patch, grid * patch)
