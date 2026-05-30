# BB

BB is a compact PyTorch repo for transformer vision experiments:

- ViT classification on CIFAR-10 or CIFAR-100.
- ViT classification on ImageNet-1k with a ViT-paper-style scratch recipe.
- Class-conditional DiT diffusion for CIFAR image generation.

The repo is intentionally small: YAML configs, reusable CIFAR dataloaders, one ViT,
one DiT, DDPM training, DDIM sampling, EMA checkpoints, and CPU-friendly smoke tests.

## Layout

```text
BB/
  bb/
    data.py          CIFAR-10/CIFAR-100 and synthetic smoke datasets
    diffusion.py     Gaussian diffusion and DDIM sampler
    train_vit.py     ViT classification training loop
    train_dit.py     DiT diffusion training loop
    sample_dit.py    checkpoint sampling entrypoint
    models/
      vit.py
      dit.py
      ema.py
  configs/
    cifar10_vit.yaml
    cifar100_vit.yaml
    cifar10_dit.yaml
    cifar100_dit.yaml
    imagenet_vit_b16_paper_scratch.yaml
    smoke_vit.yaml
    smoke_imagenet_vit.yaml
    smoke_dit.yaml
  scripts/
    train_vit.py
    train_dit.py
    sample_dit.py
    report_vit_metrics.py
    slurm_imagenet_vit_mi2104x.sbatch
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Smoke Tests

These use synthetic 32x32 tensors, so they do not download CIFAR.

```bash
python scripts/train_vit.py --config configs/smoke_vit.yaml
python scripts/train_vit.py --config configs/smoke_imagenet_vit.yaml
python scripts/train_dit.py --config configs/smoke_dit.yaml
```

## Train ViT Classification

```bash
python scripts/train_vit.py --config configs/cifar10_vit.yaml
python scripts/train_vit.py --config configs/cifar100_vit.yaml
```

Useful overrides:

```bash
python scripts/train_vit.py --config configs/cifar10_vit.yaml \
  --override data.batch_size=256 \
  --override model.depth=8 \
  --override optim.lr=0.0008
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_vit.py --config configs/cifar10_vit.yaml
```

Checkpoints are written under `run.output_dir`, with `best.pt`, `last.pt`, and
`config.resolved.json`. ViT training also writes `metrics.jsonl` with train loss,
validation loss, top-1, top-5, and best top-1.

## Train ViT on ImageNet

The ImageNet loader expects the usual ImageFolder layout:

```text
/path/to/imagenet/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
```

The paper-style ImageNet scratch config uses ViT-B/16 at 224px, Adam, beta1
0.9, beta2 0.999, cosine decay, 10k warmup steps, weight decay 0.3, dropout 0.1,
and global norm clipping at 1. With 4 GPUs, the default per-device batch size 64
and `grad_accum_steps: 16` gives an effective global batch size of 4096.

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_vit.py \
  --config configs/imagenet_vit_b16_paper_scratch.yaml \
  --override data.root=/path/to/imagenet \
  --override run.output_dir=checkpoints/imagenet_vit_b16
```

On amdhpc `mi2104x`, submit the included Slurm script after editing the ImageNet
path:

```bash
sbatch scripts/slurm_imagenet_vit_mi2104x.sbatch
```

The headline report should come from the best validation record in
`checkpoints/imagenet_vit_b16/metrics.jsonl`, using `acc1` and `acc5`.

```bash
python scripts/report_vit_metrics.py checkpoints/imagenet_vit_b16/metrics.jsonl
```

## Train DiT Image Generation

```bash
python scripts/train_dit.py --config configs/cifar10_dit.yaml
python scripts/train_dit.py --config configs/cifar100_dit.yaml
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_dit.py --config configs/cifar10_dit.yaml
```

The DiT trains directly in pixel space on CIFAR images normalized to `[-1, 1]`.
It uses class conditioning plus classifier-free label dropout. Checkpoints include
model weights, EMA weights, optimizer state, scaler state, and diffusion buffers.

## Sample DiT Checkpoints

```bash
python scripts/sample_dit.py \
  --config configs/cifar10_dit.yaml \
  --checkpoint checkpoints/cifar10_dit/last.pt \
  --output samples/cifar10_dit.png \
  --num-samples 64 \
  --ddim-steps 50 \
  --cfg-scale 1.5
```

To sample specific classes:

```bash
python scripts/sample_dit.py \
  --config configs/cifar10_dit.yaml \
  --checkpoint checkpoints/cifar10_dit/last.pt \
  --labels 0,1,2,3,4,5,6,7,8,9
```

## Practical Notes

- CIFAR-10 and CIFAR-100 are both supported by setting `data.dataset`.
- ViT inputs use standard CIFAR normalization. DiT inputs use `[-1, 1]`.
- The default DiT config is a reasonable small baseline, not an FID-tuned recipe.
- Use `run.precision=fp32` on CPU. Use `bf16` or `fp16` on recent GPUs.
- `--override a.b=value` works for all config keys.
