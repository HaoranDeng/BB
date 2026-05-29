#!/usr/bin/env bash
#SBATCH --job-name=bb-check-env
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python
print_bb_env

"$BB_PYTHON" - <<'PY'
import importlib
mods = [
    "torch",
    "yaml",
    "huggingface_hub",
    "img2dataset",
    "webdataset",
    "diffusers",
    "transformers",
    "pyarrow",
]
for name in mods:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")
PY
