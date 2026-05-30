from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import apply_dotlist_overrides, load_config
from src.data import dataset_num_classes
from src.diffusion import GaussianDiffusion
from src.models import EMA, ClassConditionalDiT
from src.utils import save_image_grid, set_seed


def parse_labels(
    raw: str | None,
    num_samples: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    if raw:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
        if not values:
            raise ValueError("--labels was provided but no labels were parsed.")
        labels = torch.tensor(values, device=device, dtype=torch.long)
        if labels.numel() < num_samples:
            repeats = (num_samples + labels.numel() - 1) // labels.numel()
            labels = labels.repeat(repeats)
        return labels[:num_samples]
    return torch.arange(num_samples, device=device, dtype=torch.long) % num_classes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample images from a trained CIFAR DiT checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="samples/dit_grid.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated class ids. Repeated if needed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = apply_dotlist_overrides(load_config(args.config), args.override)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset_num_classes(str(config["data"].get("dataset", "cifar10")))
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = num_classes
    model = ClassConditionalDiT(**model_cfg).to(device)
    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if not args.no_ema and "ema" in checkpoint:
        ema = EMA(model, decay=float(config["optim"].get("ema_decay", 0.9999)))
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        ema.copy_to(model)
    if "diffusion" in checkpoint:
        diffusion.load_state_dict(checkpoint["diffusion"])

    model.eval()
    labels = parse_labels(args.labels, args.num_samples, num_classes, device)
    images = diffusion.ddim_sample(
        model,
        shape=(
            args.num_samples,
            3,
            int(config["model"].get("img_size", 32)),
            int(config["model"].get("img_size", 32)),
        ),
        labels=labels,
        device=device,
        steps=args.ddim_steps,
        cfg_scale=args.cfg_scale,
        progress=True,
    )
    save_image_grid(images, Path(args.output), nrow=8)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
