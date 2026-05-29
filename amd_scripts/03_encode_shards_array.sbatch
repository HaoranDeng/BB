#!/usr/bin/env bash
#SBATCH --job-name=bb-dc-encode
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:55:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python

task_id="${SLURM_ARRAY_TASK_ID:-0}"
shard_list="$BB_MANIFEST_DIR/encode_task_${task_id}.txt"
if [[ ! -s "$shard_list" ]]; then
  echo "Missing shard list: $shard_list" >&2
  exit 2
fi

print_bb_env
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=$task_id"
echo "shard_list=$shard_list"

"$BB_PYTHON" scripts/encode_shards.py \
  --input-shards "@$shard_list" \
  --output-dir "$BB_ENCODED_DIR" \
  --output-prefix "part_${task_id}_" \
  --image-size "$BB_IMAGE_SIZE" \
  --batch-size "$BB_ENCODE_BATCH_SIZE" \
  --workers "$BB_ENCODE_WORKERS" \
  --samples-per-shard "$BB_ENCODE_SAMPLES_PER_SHARD" \
  --precision bf16
