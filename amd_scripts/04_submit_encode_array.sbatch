#!/usr/bin/env bash
#SBATCH --job-name=bb-submit-encode
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

mapfile -t shards < <(find "$BB_WDS_DIR" -type f -name '*.tar' | sort)
count="${#shards[@]}"
if [[ "$count" -eq 0 ]]; then
  echo "No WebDataset tar shards found in $BB_WDS_DIR" >&2
  exit 2
fi

rm -f "$BB_MANIFEST_DIR"/encode_task_*.txt
task=0
idx=0
while [[ "$idx" -lt "$count" ]]; do
  list="$BB_MANIFEST_DIR/encode_task_${task}.txt"
  : > "$list"
  for ((j = 0; j < BB_ENCODE_SHARDS_PER_TASK && idx < count; j++, idx++)); do
    printf '%s\n' "${shards[$idx]}" >> "$list"
  done
  task=$((task + 1))
done

array_max=$((task - 1))
if [[ "$task" -gt "$BB_MAX_ENCODE_TASKS" ]]; then
  echo "Refusing to submit $task encode tasks; raise BB_MAX_ENCODE_TASKS if intended." >&2
  exit 2
fi

echo "webdataset_shards=$count"
echo "encode_tasks=$task"
echo "shards_per_task=$BB_ENCODE_SHARDS_PER_TASK"

job_id="$(sbatch --parsable \
  --partition="$BB_GPU_PARTITION" \
  --array="0-${array_max}%${BB_ENCODE_ARRAY_CONCURRENCY}" \
  --export=ALL \
  amd_scripts/03_encode_shards_array.sbatch)"
echo "submitted_encode_array_job_id=$job_id"
printf '%s\n' "$job_id" > "$BB_MANIFEST_DIR/last_encode_array_job_id.txt"
