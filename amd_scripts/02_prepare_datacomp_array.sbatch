#!/usr/bin/env bash
#SBATCH --job-name=bb-dc-wds
#SBATCH --partition=mi2101x
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
task_list="$BB_MANIFEST_DIR/prepare_task_${task_id}.txt"
task_metadata_dir="$BB_MANIFEST_DIR/prepare_task_${task_id}_metadata"
task_output_dir="$BB_WDS_DIR/task_${task_id}"

if [[ ! -s "$task_list" ]]; then
  echo "Missing metadata list: $task_list" >&2
  exit 2
fi

rm -rf "$task_metadata_dir"
mkdir -p "$task_metadata_dir" "$task_output_dir"

idx=0
while IFS= read -r parquet_path; do
  [[ -n "$parquet_path" ]] || continue
  if [[ ! -f "$parquet_path" ]]; then
    echo "Missing parquet file: $parquet_path" >&2
    exit 2
  fi
  ln -s "$parquet_path" "$task_metadata_dir/$(printf '%05d' "$idx")_$(basename "$parquet_path")"
  idx=$((idx + 1))
done < "$task_list"

print_bb_env
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=$task_id"
echo "task_list=$task_list"
echo "task_output_dir=$task_output_dir"

"$BB_PYTHON" scripts/prepare_datacomp.py \
  --metadata-dir "$task_metadata_dir" \
  --output-dir "$task_output_dir" \
  --image-size "$BB_IMAGE_SIZE" \
  --processes "$BB_PREPARE_PROCESSES" \
  --threads "$BB_PREPARE_THREADS"
