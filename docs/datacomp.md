# DataComp-1B Notes

DataComp-1B is a curated subset from the DataComp xlarge pool. In practice you
will handle it in three stages:

1. Metadata: Parquet files containing URLs, captions, and filtering metadata.
2. Image acquisition: download URLs into WebDataset tar shards.
3. Feature encoding: convert images/captions into latent and text-embedding
   shards used by BB's trainer.

The repo keeps stage 3 separate from training because MI300 time is more useful
when it is spent on DiT updates rather than repeatedly running frozen encoders.

## Suggested First Slice

For a first training recipe check:

- Download 16-64 metadata files.
- Download enough images for 1M-5M successful samples.
- Encode 512px latents.
- Train `configs/pretrain_datacomp_0_6b.yaml` for 2k-5k steps.

If loss curves, samples, and throughput look sane, scale the shard count until
the training stream covers the target token budget.

## Quality Filters Worth Adding

The starter scripts do only light filtering. Before a serious run, add:

- Minimum caption length and language filtering.
- Image size/aspect-ratio filtering.
- NSFW and watermark filtering.
- Deduplication by URL and perceptual hash.
- Aesthetic or CLIP-score based sampling.

## Sources

- https://github.com/mlfoundations/datacomp
- https://huggingface.co/datasets/mlfoundations/datacomp_1b
