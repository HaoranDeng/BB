from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from src.config import apply_dotlist_overrides, load_config
from src.data import build_dataloader
from src.distributed import (
    DistributedState,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_mean,
    reduce_sum,
)
from src.models import VisionTransformer
from src.utils import (
    accuracy,
    append_jsonl,
    autocast_dtype,
    count_parameters,
    raw_model,
    save_json,
    scheduled_lr,
    set_lr,
    set_seed,
)


def build_model(config: dict[str, Any], num_classes: int, state: DistributedState) -> nn.Module:
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = num_classes
    model = VisionTransformer(**model_cfg).to(state.device)
    if bool(config["run"].get("compile", False)):
        model = torch.compile(model)
    if state.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[state.local_rank] if state.device.type == "cuda" else None,
            output_device=state.local_rank if state.device.type == "cuda" else None,
        )
    return model


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    optim_cfg = config["optim"]
    name = str(optim_cfg.get("name", "adamw")).lower()
    betas = (
        float(optim_cfg.get("beta1", 0.9)),
        float(optim_cfg.get("beta2", 0.999)),
    )
    weight_decay = float(optim_cfg.get("weight_decay", 0.05))
    lr = float(optim_cfg["lr"])
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def init_wandb(
    config: dict[str, Any],
    output_dir: Path,
    params: int,
    world_size: int,
) -> Any | None:
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)) or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging is enabled, but wandb is not installed.") from exc

    run_config = {
        **config,
        "derived": {
            "parameters": params,
            "world_size": world_size,
            "effective_batch_size": int(config["data"]["batch_size"])
            * world_size
            * int(config["optim"].get("grad_accum_steps", 1)),
        },
    }
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=str(wandb_cfg.get("project", "bb-imagenet-vit")),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        mode=str(wandb_cfg.get("mode", "online")),
        dir=str(wandb_dir),
        config=run_config,
        resume=str(wandb_cfg.get("resume", "allow")),
    )


def wandb_log(run: Any | None, payload: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(payload, step=step)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    step: int,
    best_acc1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "best_acc1": best_acc1,
            "model": raw_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    state: DistributedState,
    precision: str,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    amp_dtype = autocast_dtype(precision)
    loss_sum = torch.zeros((), device=state.device)
    top1_sum = torch.zeros((), device=state.device)
    top5_sum = torch.zeros((), device=state.device)
    total = torch.zeros((), device=state.device)
    top5_k = min(5, num_classes)

    for images, labels in loader:
        images = images.to(state.device, non_blocking=True)
        labels = labels.to(state.device, non_blocking=True)
        with torch.autocast(
            device_type=state.device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None and state.device.type == "cuda",
        ):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels)
        batch = labels.shape[0]
        top1, top5 = accuracy(logits.float(), labels, topk=(1, top5_k))
        loss_sum += loss.detach() * batch
        top1_sum += top1
        top5_sum += top5
        total += batch

    loss_sum = reduce_sum(loss_sum)
    top1_sum = reduce_sum(top1_sum)
    top5_sum = reduce_sum(top5_sum)
    total = reduce_sum(total)
    model.train()
    return {
        "loss": float((loss_sum / total).item()),
        "acc1": float((top1_sum / total * 100).item()),
        "acc5": float((top5_sum / total * 100).item()),
    }


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
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
    barrier()

    train_bundle = build_dataloader(
        config,
        task="classification",
        train=True,
        seed=seed,
        rank=state.rank,
        world_size=state.world_size,
    )
    val_bundle = build_dataloader(
        config,
        task="classification",
        train=False,
        seed=seed,
        rank=state.rank,
        world_size=state.world_size,
    )
    model = build_model(config, train_bundle.num_classes, state)
    optimizer = build_optimizer(config, model)
    precision = str(config["run"].get("precision", "bf16"))
    amp_dtype = autocast_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and state.device.type == "cuda")

    epochs = int(config["run"].get("epochs", 100))
    grad_accum_steps = int(config["optim"].get("grad_accum_steps", 1))
    steps_per_epoch = max(1, len(train_bundle.loader) // grad_accum_steps)
    max_steps = int(config["run"].get("max_steps", epochs * steps_per_epoch))
    total_steps = min(max_steps, epochs * steps_per_epoch)
    warmup_steps = int(config["optim"].get("warmup_steps", 0))
    base_lr = float(config["optim"]["lr"])
    min_lr = float(config["optim"].get("min_lr", 0.0))
    schedule = str(config["optim"].get("schedule", "cosine"))
    grad_clip = float(config["optim"].get("grad_clip", 0.0))
    label_smoothing = float(config["optim"].get("label_smoothing", 0.0))
    log_every = int(config["run"].get("log_every", 50))
    eval_every_epochs = int(config["run"].get("eval_every_epochs", 1))
    metrics_path = output_dir / "metrics.jsonl"

    if is_main_process():
        params = count_parameters(raw_model(model))
        print(f"ViT params={params / 1e6:.2f}M device={state.device} world_size={state.world_size}")
    else:
        params = 0
    wandb_run = init_wandb(config, output_dir, params, state.world_size)

    global_step = 0
    best_acc1 = 0.0
    model.train()
    for epoch in range(epochs):
        if train_bundle.sampler is not None:
            train_bundle.sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        accum_loss = torch.zeros((), device=state.device)
        for images, labels in train_bundle.loader:
            images = images.to(state.device, non_blocking=True)
            labels = labels.to(state.device, non_blocking=True)
            lr = scheduled_lr(
                global_step,
                total_steps,
                base_lr,
                warmup_steps,
                min_lr=min_lr,
                schedule=schedule,
            )
            set_lr(optimizer, lr)

            with torch.autocast(
                device_type=state.device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None and state.device.type == "cuda",
            ):
                logits = model(images)
                raw_loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    label_smoothing=label_smoothing,
                )
                loss = raw_loss / grad_accum_steps

            scaler.scale(loss).backward()
            accum_count += 1
            accum_loss += raw_loss.detach()

            if accum_count < grad_accum_steps:
                continue

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            train_loss = accum_loss / accum_count
            accum_count = 0
            accum_loss = torch.zeros((), device=state.device)

            if global_step % log_every == 0:
                mean_loss = reduce_mean(train_loss.clone()).item()
                if is_main_process():
                    print(
                        f"epoch={epoch + 1} step={global_step}/{total_steps} "
                        f"loss={mean_loss:.4f} lr={lr:.3e}",
                        flush=True,
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "split": "train",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": mean_loss,
                            "lr": lr,
                        },
                    )
                    wandb_log(
                        wandb_run,
                        {
                            "train/loss": mean_loss,
                            "train/lr": lr,
                            "epoch": epoch + 1,
                        },
                        global_step,
                    )
            if is_main_process() and int(config["run"].get("save_every", 0)) > 0:
                save_every = int(config["run"]["save_every"])
                if global_step % save_every == 0:
                    save_checkpoint(
                        output_dir / f"step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        scaler,
                        config,
                        epoch + 1,
                        global_step,
                        best_acc1,
                    )
            if global_step >= max_steps:
                break
        if state.distributed:
            barrier()
        if global_step >= max_steps:
            should_eval = True
        else:
            should_eval = (epoch + 1) % eval_every_epochs == 0
        if should_eval:
            metrics = evaluate(
                model,
                val_bundle.loader,
                state=state,
                precision=precision,
                num_classes=train_bundle.num_classes,
            )
            if is_main_process():
                print(
                    f"eval epoch={epoch + 1} loss={metrics['loss']:.4f} "
                    f"acc1={metrics['acc1']:.2f} acc5={metrics['acc5']:.2f}",
                    flush=True,
                )
                append_jsonl(
                    metrics_path,
                    {
                        "split": "val",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": metrics["loss"],
                        "acc1": metrics["acc1"],
                        "acc5": metrics["acc5"],
                        "best_acc1": max(best_acc1, metrics["acc1"]),
                    },
                )
                wandb_log(
                    wandb_run,
                    {
                        "val/loss": metrics["loss"],
                        "val/acc1": metrics["acc1"],
                        "val/acc5": metrics["acc5"],
                        "val/best_acc1": max(best_acc1, metrics["acc1"]),
                        "epoch": epoch + 1,
                    },
                    global_step,
                )
                if metrics["acc1"] >= best_acc1:
                    best_acc1 = metrics["acc1"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        scaler,
                        config,
                        epoch + 1,
                        global_step,
                        best_acc1,
                    )
                save_checkpoint(
                    output_dir / "last.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    epoch + 1,
                    global_step,
                    best_acc1,
                )
        if global_step >= max_steps:
            break

    barrier()
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ViT classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = apply_dotlist_overrides(load_config(args.config), args.override)
    train(config)


if __name__ == "__main__":
    main()
