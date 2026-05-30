from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[torch.Tensor]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        values.append(correct[:k].reshape(-1).float().sum())
    return values


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (base_lr - min_lr) * cosine


def linear_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * max(0.0, 1.0 - progress)


def scheduled_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr: float = 0.0,
    schedule: str = "cosine",
) -> float:
    if schedule == "cosine":
        return cosine_lr(step, total_steps, base_lr, warmup_steps, min_lr=min_lr)
    if schedule == "linear":
        return linear_lr(step, total_steps, base_lr, warmup_steps, min_lr=min_lr)
    if schedule == "constant":
        return base_lr
    raise ValueError(f"Unknown LR schedule: {schedule}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError(f"Unknown precision: {precision}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (images.detach().float().cpu().clamp(-1, 1) + 1) * 0.5
    save_image(images, path, nrow=nrow)
