from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from bb.config import apply_dotlist_overrides, load_config
from bb.data import build_dataloader
from bb.distributed import DistributedState, barrier, init_distributed, is_main_process, reduce_mean
from bb.models import EMA, ImageGenDiT, count_parameters


def cosine_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    angle = t * math.pi / 2
    return torch.cos(angle), torch.sin(angle)


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError(f"Unknown precision: {precision}")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def infinite(loader: Any) -> Any:
    while True:
        for batch in loader:
            yield batch


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def learning_rate(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw_model(model).state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def build_model(config: dict[str, Any], state: DistributedState) -> torch.nn.Module:
    model = ImageGenDiT(**config["model"]).to(state.device)
    if bool(config["run"].get("compile", False)):
        model = torch.compile(model)
    if state.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[state.local_rank] if state.device.type == "cuda" else None,
            output_device=state.local_rank if state.device.type == "cuda" else None,
        )
    return model


def train(config: dict[str, Any]) -> None:
    state = init_distributed()
    seed = int(config["run"].get("seed", 0)) + state.rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(config["run"]["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    barrier()

    dataloader = build_dataloader(config, seed=seed)
    iterator = infinite(dataloader)
    model = build_model(config, state)
    unwrapped = raw_model(model)
    ema = EMA(unwrapped, decay=float(config["optim"].get("ema_decay", 0.9999)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        betas=(float(config["optim"]["beta1"]), float(config["optim"]["beta2"])),
        weight_decay=float(config["optim"]["weight_decay"]),
    )

    precision = str(config["run"].get("precision", "bf16"))
    amp_dtype = autocast_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and state.device.type == "cuda")
    grad_accum_steps = int(config["optim"].get("grad_accum_steps", 1))
    max_steps = int(config["run"]["max_steps"])
    log_every = int(config["run"].get("log_every", 10))
    save_every = int(config["run"].get("save_every", 1000))
    warmup_steps = int(config["optim"].get("warmup_steps", 0))
    base_lr = float(config["optim"]["lr"])
    grad_clip = float(config["optim"].get("grad_clip", 0))
    min_t = float(config["diffusion"].get("min_t", 0.001))
    max_t = float(config["diffusion"].get("max_t", 0.999))

    if is_main_process():
        params = count_parameters(unwrapped)
        print(f"parameters={params / 1e9:.3f}B device={state.device} world_size={state.world_size}")

    start_time = time.time()
    model.train()
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=state.device)
        lr = learning_rate(step, base_lr, warmup_steps)
        set_lr(optimizer, lr)

        for _ in range(grad_accum_steps):
            batch = move_batch(next(iterator), state.device)
            latents = batch["latents"]
            text_embeds = batch["text_embeds"]
            batch_size = latents.shape[0]
            timesteps = torch.empty(batch_size, device=state.device).uniform_(min_t, max_t)
            noise = torch.randn_like(latents)
            alpha, sigma = cosine_alpha_sigma(timesteps)
            noisy_latents = alpha[:, None, None, None] * latents + sigma[:, None, None, None] * noise

            autocast_enabled = amp_dtype is not None and state.device.type == "cuda"
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.float32,
                enabled=autocast_enabled,
            ):
                prediction = model(noisy_latents, timesteps, text_embeds)
                loss = F.mse_loss(prediction.float(), noise.float())
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.detach()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(unwrapped)

        mean_loss = reduce_mean(step_loss.clone())
        if is_main_process() and (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"step={step + 1} loss={mean_loss.item():.6f} "
                f"lr={lr:.3e} seconds={elapsed:.1f}",
                flush=True,
            )
        if is_main_process() and (step + 1) % save_every == 0:
            save_checkpoint(
                output_dir / f"step_{step + 1:08d}.pt",
                model,
                ema,
                optimizer,
                config,
                step + 1,
            )

    if is_main_process():
        save_checkpoint(output_dir / "last.pt", model, ema, optimizer, config, max_steps)
    barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotlist override, e.g. optim.lr=3e-4",
    )
    args = parser.parse_args()
    config = apply_dotlist_overrides(load_config(args.config), args.override)
    train(config)


if __name__ == "__main__":
    main()
