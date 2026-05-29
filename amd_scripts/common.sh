#!/usr/bin/env bash
set -euo pipefail

export BB_REPO_DIR="${BB_REPO_DIR:-$HOME/workspace/BB}"
export BB_DATA_ROOT="${BB_DATA_ROOT:-/work1/jasoncong/denghaoran/BB/data}"
export BB_LOG_DIR="${BB_LOG_DIR:-$BB_REPO_DIR/logs/slurm}"
export BB_METADATA_DIR="${BB_METADATA_DIR:-$BB_DATA_ROOT/datacomp_1b/metadata}"
export BB_WDS_DIR="${BB_WDS_DIR:-$BB_DATA_ROOT/datacomp_1b/wds_512}"
export BB_ENCODED_DIR="${BB_ENCODED_DIR:-$BB_DATA_ROOT/datacomp_1b/encoded_512}"
export BB_MANIFEST_DIR="${BB_MANIFEST_DIR:-$BB_DATA_ROOT/manifests}"

export BB_DATACOMP_REPO_ID="${BB_DATACOMP_REPO_ID:-mlfoundations/datacomp_1b}"
export BB_DATACOMP_MAX_FILES="${BB_DATACOMP_MAX_FILES:-64}"
export BB_IMAGE_SIZE="${BB_IMAGE_SIZE:-512}"
export BB_PREPARE_PROCESSES="${BB_PREPARE_PROCESSES:-64}"
export BB_PREPARE_THREADS="${BB_PREPARE_THREADS:-128}"
export BB_ENCODE_BATCH_SIZE="${BB_ENCODE_BATCH_SIZE:-32}"
export BB_ENCODE_WORKERS="${BB_ENCODE_WORKERS:-8}"
export BB_ENCODE_SHARDS_PER_TASK="${BB_ENCODE_SHARDS_PER_TASK:-8}"
export BB_ENCODE_SAMPLES_PER_SHARD="${BB_ENCODE_SAMPLES_PER_SHARD:-2048}"
export BB_MAX_ENCODE_TASKS="${BB_MAX_ENCODE_TASKS:-512}"

export BB_CPU_PARTITION="${BB_CPU_PARTITION:-mi3001x}"
export BB_GPU_PARTITION="${BB_GPU_PARTITION:-mi3008x}"
export BB_DEVEL_PARTITION="${BB_DEVEL_PARTITION:-devel}"

export BB_PYTHON="${BB_PYTHON:-python}"
export BB_CONDA_ENV="${BB_CONDA_ENV:-}"
export BB_VENV="${BB_VENV:-$BB_REPO_DIR/.venv}"

export HF_HOME="${HF_HOME:-$BB_DATA_ROOT/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$BB_DATA_ROOT/python_user_base}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$BB_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

activate_bb_python() {
  if [[ -n "$BB_CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "$BB_CONDA_ENV"
    else
      echo "BB_CONDA_ENV=$BB_CONDA_ENV but conda is not on PATH" >&2
      exit 2
    fi
  elif [[ -f "$BB_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$BB_VENV/bin/activate"
  fi

  if ! command -v "$BB_PYTHON" >/dev/null 2>&1; then
    echo "Python command not found: $BB_PYTHON" >&2
    exit 2
  fi
}

prepare_bb_dirs() {
  mkdir -p "$BB_DATA_ROOT" "$BB_LOG_DIR" "$BB_METADATA_DIR" "$BB_WDS_DIR" \
    "$BB_ENCODED_DIR" "$BB_MANIFEST_DIR" "$HF_HOME" "$PYTHONUSERBASE"
}

print_bb_env() {
  cat <<EOF
BB_REPO_DIR=$BB_REPO_DIR
BB_DATA_ROOT=$BB_DATA_ROOT
BB_METADATA_DIR=$BB_METADATA_DIR
BB_WDS_DIR=$BB_WDS_DIR
BB_ENCODED_DIR=$BB_ENCODED_DIR
BB_DATACOMP_MAX_FILES=$BB_DATACOMP_MAX_FILES
BB_IMAGE_SIZE=$BB_IMAGE_SIZE
BB_CPU_PARTITION=$BB_CPU_PARTITION
BB_GPU_PARTITION=$BB_GPU_PARTITION
BB_PYTHON=$BB_PYTHON
BB_CONDA_ENV=$BB_CONDA_ENV
BB_VENV=$BB_VENV
PYTHONUSERBASE=$PYTHONUSERBASE
EOF
}
