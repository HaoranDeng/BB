# BB

BB is a compact PyTorch repo for transformer vision experiments:

- ViT classification on CIFAR-10 or CIFAR-100.
- ViT classification on ImageNet-1k with a ViT-paper-style scratch recipe.
- Class-conditional DiT diffusion for CIFAR image generation.

The repo is intentionally small: YAML configs, reusable CIFAR dataloaders, one ViT,
one DiT, DDPM training, DDIM sampling, EMA checkpoints, and CPU-friendly smoke tests.
Large files are expected to live under `$WORK`: datasets under `$WORK/bb/data`,
and checkpoints, samples, and W&B files under `$WORK/bb/checkpoints`.
Config files support environment variables such as `${WORK}`.

## Layout

```text
BB/
  attention/
    standard.py      scaled dot-product self-attention
    linear.py        kernelized linear attention
    monarch.py       two-stage block/global Monarch-style attention
  src/
    data.py          CIFAR-10/CIFAR-100 and synthetic smoke datasets
    diffusion.py     Gaussian diffusion and DDIM sampler
    train_vit.py     ViT classification training loop
    train_dit.py     DiT diffusion training loop
    sample_dit.py    checkpoint sampling entrypoint
    models/
      vit.py
      dit.py
      ema.py
  tools/
    train_vit.py
    train_dit.py
    sample_dit.py
    report_vit_metrics.py
  scripts/
    train_cifar_vit.sh
    train_cifar_dit.sh
    train_imagenet_vit_mi2104x.sh
  configs/
    cifar10_vit.yaml
    cifar100_vit.yaml
    cifar10_dit.yaml
    cifar100_dit.yaml
    imagenet_vit_b16_paper_scratch.yaml
    smoke_vit.yaml
    smoke_imagenet_vit.yaml
    smoke_dit.yaml
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
python tools/train_vit.py --config configs/smoke_vit.yaml
python tools/train_vit.py --config configs/smoke_imagenet_vit.yaml
python tools/train_dit.py --config configs/smoke_dit.yaml
```

## Train ViT Classification

```bash
python tools/train_vit.py --config configs/cifar10_vit.yaml
python tools/train_vit.py --config configs/cifar100_vit.yaml
```

Useful overrides:

```bash
python tools/train_vit.py --config configs/cifar10_vit.yaml \
  --override data.batch_size=256 \
  --override model.depth=8 \
  --override optim.lr=0.0008
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=4 tools/train_vit.py --config configs/cifar10_vit.yaml
```

Checkpoints are written under `run.output_dir`, with `best.pt`, `last.pt`, and
`config.resolved.json`. ViT training also writes `metrics.jsonl` with train loss,
validation loss, top-1, top-5, and best top-1.

## Attention Variants

Attention implementations live in the top-level `attention/` package. Each file
contains one implementation:

- `attention/standard.py`: standard scaled dot-product attention.
- `attention/linear.py`: ELU feature-map linear attention.
- `attention/monarch.py`: two-stage Monarch-style block/global attention.

Switch variants from YAML or command line:

```bash
python tools/train_vit.py --config configs/cifar10_vit.yaml \
  --override model.attention=linear
```

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
torchrun --standalone --nproc_per_node=4 tools/train_vit.py \
  --config configs/imagenet_vit_b16_paper_scratch.yaml \
  --override data.root="${WORK}/bb/data/imagenet" \
  --override run.output_dir="${WORK}/bb/checkpoints/imagenet_vit_b16"
```

The headline report should come from the best validation record in
`$WORK/bb/checkpoints/imagenet_vit_b16/metrics.jsonl`, using `acc1` and `acc5`.

```bash
python tools/report_vit_metrics.py "$WORK/bb/checkpoints/imagenet_vit_b16/metrics.jsonl"
```

## Train DiT Image Generation

```bash
python tools/train_dit.py --config configs/cifar10_dit.yaml
python tools/train_dit.py --config configs/cifar100_dit.yaml
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=4 tools/train_dit.py --config configs/cifar10_dit.yaml
```

The DiT trains directly in pixel space on CIFAR images normalized to `[-1, 1]`.
It uses class conditioning plus classifier-free label dropout. Checkpoints include
model weights, EMA weights, optimizer state, scaler state, and diffusion buffers.

## Sample DiT Checkpoints

```bash
python tools/sample_dit.py \
  --config configs/cifar10_dit.yaml \
  --checkpoint "$WORK/bb/checkpoints/cifar10_dit/last.pt" \
  --output "$WORK/bb/checkpoints/samples/cifar10_dit.png" \
  --num-samples 64 \
  --ddim-steps 50 \
  --cfg-scale 1.5
```

To sample specific classes:

```bash
python tools/sample_dit.py \
  --config configs/cifar10_dit.yaml \
  --checkpoint "$WORK/bb/checkpoints/cifar10_dit/last.pt" \
  --labels 0,1,2,3,4,5,6,7,8,9
```

## Practical Notes

- CIFAR-10 and CIFAR-100 are both supported by setting `data.dataset`.
- ViT inputs use standard CIFAR normalization. DiT inputs use `[-1, 1]`.
- The default DiT config is a reasonable small baseline, not an FID-tuned recipe.
- Use `run.precision=fp32` on CPU. Use `bf16` or `fp16` on recent GPUs.
- `--override a.b=value` works for all config keys.
