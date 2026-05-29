#!/usr/bin/env bash
#SBATCH --job-name=bb-submit-wds
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs

mapfile -t parquet_files < <(find "$BB_METADATA_DIR" -type f -name '*.parquet' | sort)
count="${#parquet_files[@]}"
if [[ "$count" -eq 0 ]]; then
  echo "No metadata parquet files found in $BB_METADATA_DIR" >&2
  exit 2
fi

rm -f "$BB_MANIFEST_DIR"/prepare_task_*.txt
task=0
idx=0
while [[ "$idx" -lt "$count" ]]; do
  list="$BB_MANIFEST_DIR/prepare_task_${task}.txt"
  : > "$list"
  for ((j = 0; j < BB_PREPARE_FILES_PER_TASK && idx < count; j++, idx++)); do
    printf '%s\n' "${parquet_files[$idx]}" >> "$list"
  done
  task=$((task + 1))
done

array_max=$((task - 1))
if [[ "$task" -gt "$BB_MAX_PREPARE_TASKS" ]]; then
  echo "Refusing to submit $task prepare tasks; raise BB_MAX_PREPARE_TASKS if intended." >&2
  exit 2
fi

echo "metadata_files=$count"
echo "prepare_tasks=$task"
echo "files_per_task=$BB_PREPARE_FILES_PER_TASK"

prepare_array_job="$(sbatch --parsable \
  --partition="$BB_CPU_PARTITION" \
  --array="0-${array_max}" \
  --export=ALL \
  amd_scripts/02_prepare_datacomp_array.sbatch)"

encode_submit_job="$(sbatch --parsable \
  --partition="$BB_DEVEL_PARTITION" \
  --dependency="afterok:$prepare_array_job" \
  --export=ALL \
  amd_scripts/04_submit_encode_array.sbatch)"

echo "submitted_prepare_array_job_id=$prepare_array_job"
echo "submitted_encode_submit_job_id=$encode_submit_job"
printf '%s\n' "$prepare_array_job" > "$BB_MANIFEST_DIR/last_prepare_array_job_id.txt"
printf '%s\n' "$encode_submit_job" > "$BB_MANIFEST_DIR/last_encode_submit_job_id.txt"
