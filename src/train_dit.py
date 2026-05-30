from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.config import apply_dotlist_overrides, load_config
from src.data import build_dataloader
from src.diffusion import GaussianDiffusion
from src.distributed import (
    DistributedState,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_mean,
)
from src.models import EMA, ClassConditionalDiT
from src.utils import (
    autocast_dtype,
    cosine_lr,
    count_parameters,
    raw_model,
    save_image_grid,
    save_json,
    set_lr,
    set_seed,
)


def infinite(loader: torch.utils.data.DataLoader, sampler: Any = None) -> Iterator:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def build_model(config: dict[str, Any], num_classes: int, state: DistributedState) -> nn.Module:
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = num_classes
    model = ClassConditionalDiT(**model_cfg).to(state.device)
    if bool(config["run"].get("compile", False)):
        model = torch.compile(model)
    if state.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[state.local_rank] if state.device.type == "cuda" else None,
            output_device=state.local_rank if state.device.type == "cuda" else None,
        )
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    diffusion: GaussianDiffusion,
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
            "scaler": scaler.state_dict(),
            "diffusion": diffusion.state_dict(),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def write_samples(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    output_dir: Path,
    step: int,
    num_classes: int,
    state: DistributedState,
    config: dict[str, Any],
) -> None:
    if not is_main_process():
        return
    sample_cfg = config.get("sample", {})
    num_samples = int(sample_cfg.get("num_samples", 64))
    img_size = int(config["model"].get("img_size", 32))
    labels = torch.arange(num_samples, device=state.device) % num_classes
    raw_model(model).eval()
    images = diffusion.ddim_sample(
        raw_model(model),
        shape=(num_samples, 3, img_size, img_size),
        labels=labels,
        device=state.device,
        steps=int(sample_cfg.get("ddim_steps", 50)),
        cfg_scale=float(sample_cfg.get("cfg_scale", 1.0)),
        progress=False,
    )
    raw_model(model).train()
    save_image_grid(
        images,
        output_dir / "samples" / f"step_{step:08d}.png",
        nrow=int(sample_cfg.get("nrow", 8)),
    )


def train(config: dict[str, Any]) -> None:
    state = init_distributed()
    seed = int(config["run"].get("seed", 0)) + state.rank
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(config["run"]["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "config.resolved.json", config)
    barrier()

    train_bundle = build_dataloader(
        config,
        task="generation",
        train=True,
        seed=seed,
        rank=state.rank,
        world_size=state.world_size,
    )
    iterator = infinite(train_bundle.loader, train_bundle.sampler)
    model = build_model(config, train_bundle.num_classes, state)
    diffusion = GaussianDiffusion(**config["diffusion"]).to(state.device)
    ema = EMA(raw_model(model), decay=float(config["optim"].get("ema_decay", 0.9999)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        betas=(
            float(config["optim"].get("beta1", 0.9)),
            float(config["optim"].get("beta2", 0.999)),
        ),
        weight_decay=float(config["optim"].get("weight_decay", 0.0)),
    )
    precision = str(config["run"].get("precision", "bf16"))
    amp_dtype = autocast_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and state.device.type == "cuda")

    max_steps = int(config["run"]["max_steps"])
    warmup_steps = int(config["optim"].get("warmup_steps", 0))
    base_lr = float(config["optim"]["lr"])
    min_lr = float(config["optim"].get("min_lr", 0.0))
    grad_accum_steps = int(config["optim"].get("grad_accum_steps", 1))
    grad_clip = float(config["optim"].get("grad_clip", 0.0))
    log_every = int(config["run"].get("log_every", 50))
    save_every = int(config["run"].get("save_every", 1000))
    sample_every = int(config["run"].get("sample_every", 0))

    if is_main_process():
        params = count_parameters(raw_model(model))
        print(f"DiT params={params / 1e6:.2f}M device={state.device} world_size={state.world_size}")

    model.train()
    for step in range(max_steps):
        lr = cosine_lr(step, max_steps, base_lr, warmup_steps, min_lr=min_lr)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=state.device)

        for _ in range(grad_accum_steps):
            images, labels = next(iterator)
            images = images.to(state.device, non_blocking=True)
            labels = labels.to(state.device, non_blocking=True)
            with torch.autocast(
                device_type=state.device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None and state.device.type == "cuda",
            ):
                loss = diffusion.training_loss(model, images, labels) / grad_accum_steps
            scaler.scale(loss).backward()
            step_loss += loss.detach()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(raw_model(model))

        step_id = step + 1
        if step_id % log_every == 0:
            mean_loss = reduce_mean(step_loss.clone()).item()
        else:
            mean_loss = 0.0
        if is_main_process() and step_id % log_every == 0:
            print(f"step={step_id}/{max_steps} loss={mean_loss:.4f} lr={lr:.3e}", flush=True)
        if sample_every > 0 and step_id % sample_every == 0:
            write_samples(
                model,
                diffusion,
                output_dir,
                step_id,
                train_bundle.num_classes,
                state,
                config,
            )
        if is_main_process() and step_id % save_every == 0:
            save_checkpoint(
                output_dir / f"step_{step_id:08d}.pt",
                model,
                ema,
                optimizer,
                scaler,
                diffusion,
                config,
                step_id,
            )

    if is_main_process():
        save_checkpoint(
            output_dir / "last.pt",
            model,
            ema,
            optimizer,
            scaler,
            diffusion,
            config,
            max_steps,
        )
    barrier()
    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a class-conditional DiT on CIFAR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = apply_dotlist_overrides(load_config(args.config), args.override)
    train(config)


if __name__ == "__main__":
    main()
