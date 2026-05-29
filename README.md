# BB

BB is a small, hackable ImageGen pretraining repo for 0.6B-ish diffusion-transformer
experiments on DataComp-1B. It is intentionally boring in the right places:

- DataComp-1B metadata and image download helpers.
- Offline VAE/text-encoder feature encoding into `.pt` shards.
- A text-conditioned DiT training loop that can run a synthetic smoke test first.
- A 0.6B-class config aimed at 8x MI300X/MI300A style boxes.

I could not access `HaoranDeng/AA` from this environment, so this is a clean BB
starter shaped around your ImageGen pretraining goal. Once AA is available as a
zip or through GitHub authorization, the repo style can be aligned more closely.

## Layout

```text
BB/
  bb/
    data/          encoded shard and synthetic datasets
    models/        DiT backbone and EMA helper
    train.py       training loop
    config.py      YAML config loader
    distributed.py distributed utilities
  configs/
    pretrain_datacomp_0_6b.yaml
    smoke.yaml
  scripts/
    download_datacomp_metadata.py
    prepare_datacomp.py
    encode_shards.py
    train.py
  examples/
    train_mi300_8gpu.sh
  docs/
    datacomp.md
```

## Install

Use the ROCm PyTorch wheel that matches your driver/runtime first, then install
the repo dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

For MI300 training, verify BF16 matmul and RCCL before launching a long run.

## Smoke Test

This does not need DataComp. It uses random latents and text embeddings to check
the model, loss, optimizer, checkpointing, and distributed code path.

```bash
python scripts/train.py --config configs/smoke.yaml
```

## DataComp-1B Flow

DataComp-1B is released as Parquet metadata on Hugging Face by ML Foundations.
The original DataComp repo also ships download helpers for fetching images from
the URLs in those metadata files.

1. Download a metadata slice:

```bash
python scripts/download_datacomp_metadata.py \
  --repo-id mlfoundations/datacomp_1b \
  --local-dir data/datacomp_1b/metadata \
  --max-files 64
```

2. Convert metadata rows into WebDataset image/text tar shards:

```bash
python scripts/prepare_datacomp.py \
  --metadata-dir data/datacomp_1b/metadata \
  --output-dir data/datacomp_1b/wds_512 \
  --image-size 512 \
  --processes 64
```

3. Encode images and captions offline:

```bash
python scripts/encode_shards.py \
  --input-shards "data/datacomp_1b/wds_512/{00000..00127}.tar" \
  --output-dir data/datacomp_1b/encoded_512 \
  --vae stabilityai/sd-vae-ft-ema \
  --text-encoder google/t5-v1_1-base \
  --batch-size 32 \
  --max-text-tokens 64
```

4. Train the 0.6B-class DiT:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/pretrain_datacomp_0_6b.yaml
```

The 10B-token experiment target is roughly:

- 512px with 32x VAE latent grid: `32 x 32 = 1024` latent positions/image if
  using 16x compression, or `16 x 16 = 256` with 32x compression.
- With the default `latent_size: 32` and `patch_size: 2`, the transformer sees
  `16 x 16 = 256` visual tokens/sample.
- So 10B transformer visual tokens is about 39M image samples.

## Notes

- The default model uses cross-attention over text encoder hidden states and
  predicts diffusion noise in latent space.
- `configs/pretrain_datacomp_0_6b.yaml` is a starting recipe, not a final
  scaling-law answer. Expect to tune batch size, LR, warmup, and resolution mix.
- For a first real run, use a small metadata slice and train for a few thousand
  steps before committing to the full 10B-token budget.

## References

- DataComp repo: https://github.com/mlfoundations/datacomp
- DataComp-1B Hugging Face dataset: https://huggingface.co/datasets/mlfoundations/datacomp_1b
