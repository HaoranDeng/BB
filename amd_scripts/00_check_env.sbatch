#!/usr/bin/env bash
#SBATCH --job-name=bb-check-env
#SBATCH --partition=mi2101x
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
    "torchvision",
    "yaml",
    "huggingface_hub",
    "img2dataset",
    "webdataset",
    "diffusers",
    "transformers",
    "pyarrow",
    "PIL",
    "tqdm",
    "safetensors",
    "accelerate",
    "numpy",
]
missing = []
for name in mods:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        missing.append(name)
        print(f"MISSING {name}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit(f"Missing Python modules: {', '.join(missing)}")
PY
