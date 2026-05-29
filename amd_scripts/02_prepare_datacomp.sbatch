#!/usr/bin/env bash
#SBATCH --job-name=bb-dc-wds
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python
print_bb_env

"$BB_PYTHON" scripts/prepare_datacomp.py \
  --metadata-dir "$BB_METADATA_DIR" \
  --output-dir "$BB_WDS_DIR" \
  --image-size "$BB_IMAGE_SIZE" \
  --processes "$BB_PREPARE_PROCESSES" \
  --threads "$BB_PREPARE_THREADS"
