#!/usr/bin/env bash
#SBATCH --job-name=bb-dc-meta
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python
print_bb_env

"$BB_PYTHON" scripts/download_datacomp_metadata.py \
  --repo-id "$BB_DATACOMP_REPO_ID" \
  --local-dir "$BB_METADATA_DIR" \
  --max-files "$BB_DATACOMP_MAX_FILES"
