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

IMAGENET_ROOT="${IMAGENET_ROOT:-/path/to/imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/imagenet_vit_b16_mi2104x}"

python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train_vit.py \
  --config configs/imagenet_vit_b16_paper_scratch.yaml \
  --override data.root="${IMAGENET_ROOT}" \
  --override run.output_dir="${OUTPUT_DIR}" \
  "$@"
