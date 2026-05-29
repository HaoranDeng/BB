#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RCCL_ENABLE_SIGNALHANDLER="${RCCL_ENABLE_SIGNALHANDLER:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train.py \
  --config configs/pretrain_datacomp_0_6b.yaml
