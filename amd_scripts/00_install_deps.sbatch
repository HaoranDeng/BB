#!/usr/bin/env bash
#SBATCH --job-name=bb-install-deps
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python
print_bb_env

"$BB_PYTHON" -m pip install --user --upgrade \
  img2dataset \
  standard-imghdr
