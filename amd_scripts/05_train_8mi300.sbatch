#!/usr/bin/env bash
#SBATCH --job-name=bb-train-0p6b
#SBATCH --partition=mi3008x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=03:55:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python
print_bb_env

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RCCL_ENABLE_SIGNALHANDLER="${RCCL_ENABLE_SIGNALHANDLER:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

"$BB_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  scripts/train.py \
  --config configs/pretrain_datacomp_0_6b.yaml \
  --override "data.encoded_dir=$BB_ENCODED_DIR"
