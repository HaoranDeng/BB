#!/usr/bin/env bash
#SBATCH --job-name=bb-imagenet-vit-b16
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/imagenet-vit-b16-%j.out
#SBATCH --error=logs/imagenet-vit-b16-%j.err

set -euo pipefail

mkdir -p logs

if [[ -f /work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh ]]; then
  source /work1/jasoncong/denghaoran/miniconda3/etc/profile.d/conda.sh
  conda activate base
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

WORK_DIR="${WORK:-/work1/jasoncong/denghaoran}"
IMAGENET_ROOT="${IMAGENET_ROOT:-${WORK_DIR}/bb/data/imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/bb/checkpoints/imagenet_vit_b16_mi2104x}"
WANDB_PROJECT="${WANDB_PROJECT:-bb-imagenet-vit}"
WANDB_NAME="${WANDB_NAME:-imagenet-vit-b16-paper-scratch-mi2104x}"

python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train_vit.py \
  --config configs/imagenet_vit_b16_paper_scratch.yaml \
  --override data.root="${IMAGENET_ROOT}" \
  --override run.output_dir="${OUTPUT_DIR}" \
  --override logging.wandb.project="${WANDB_PROJECT}" \
  --override logging.wandb.name="${WANDB_NAME}" \
  "$@"
