#!/usr/bin/env bash
#SBATCH --job-name=bb-encode-chunk
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:55:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs
activate_bb_python

if [[ -z "${BB_ENCODE_CHUNK_ID:-}" ]]; then
  echo "BB_ENCODE_CHUNK_ID is required" >&2
  exit 2
fi

chunk_list="$BB_MANIFEST_DIR/encode_chunk_${BB_ENCODE_CHUNK_ID}.txt"
if [[ ! -s "$chunk_list" ]]; then
  echo "Missing encode chunk list: $chunk_list" >&2
  exit 2
fi

print_bb_env
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "BB_ENCODE_CHUNK_ID=$BB_ENCODE_CHUNK_ID"
echo "chunk_list=$chunk_list"
echo "chunk_shards=$(wc -l < "$chunk_list")"
echo "BB_ENCODE_GPUS_PER_JOB=$BB_ENCODE_GPUS_PER_JOB"

tmpdir="$(mktemp -d "$BB_MANIFEST_DIR/encode_${BB_ENCODE_CHUNK_ID}_${SLURM_JOB_ID:-manual}.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

"$BB_PYTHON" - "$chunk_list" "$tmpdir" "$BB_ENCODE_GPUS_PER_JOB" <<'PY'
import sys
from pathlib import Path

chunk_list = Path(sys.argv[1])
tmpdir = Path(sys.argv[2])
workers = int(sys.argv[3])
shards = [line.strip() for line in chunk_list.read_text().splitlines() if line.strip()]
for index in range(workers):
    part = tmpdir / f"shards_{index}.txt"
    assigned = shards[index::workers]
    part.write_text("\n".join(assigned) + ("\n" if assigned else ""))
    print(f"worker={index} shards={len(assigned)} manifest={part}")
PY

worker_pids=()
worker_count=0
workers_per_gpu="$BB_ENCODE_WORKERS"
if [[ "$BB_ENCODE_GPUS_PER_JOB" -gt 0 ]]; then
  workers_per_gpu=$((BB_ENCODE_WORKERS / BB_ENCODE_GPUS_PER_JOB))
fi
if [[ "$workers_per_gpu" -lt 1 ]]; then
  workers_per_gpu=1
fi

for worker_id in $(seq 0 $((BB_ENCODE_GPUS_PER_JOB - 1))); do
  part="$tmpdir/shards_${worker_id}.txt"
  if [[ ! -s "$part" ]]; then
    continue
  fi
  worker_log="$BB_LOG_DIR/bb-encode-chunk-${SLURM_JOB_ID:-manual}-worker${worker_id}.out"
  worker_err="$BB_LOG_DIR/bb-encode-chunk-${SLURM_JOB_ID:-manual}-worker${worker_id}.err"
  echo "starting worker=$worker_id manifest=$part log=$worker_log err=$worker_err"
  (
    export HIP_VISIBLE_DEVICES="$worker_id"
    export CUDA_VISIBLE_DEVICES="$worker_id"
    "$BB_PYTHON" scripts/encode_shards.py \
      --input-shards "@$part" \
      --output-dir "$BB_ENCODED_DIR" \
      --output-prefix "chunk_${BB_ENCODE_CHUNK_ID}_worker${worker_id}_" \
      --image-size "$BB_IMAGE_SIZE" \
      --batch-size "$BB_ENCODE_BATCH_SIZE" \
      --workers "$workers_per_gpu" \
      --samples-per-shard "$BB_ENCODE_SAMPLES_PER_SHARD" \
      --precision bf16
  ) >"$worker_log" 2>"$worker_err" &
  worker_pids+=("$!")
  worker_count=$((worker_count + 1))
done

if [[ "$worker_count" -eq 0 ]]; then
  echo "No encode workers were started" >&2
  exit 2
fi

failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more encode workers failed" >&2
  exit 1
fi

touch "$BB_MANIFEST_DIR/encode_chunk_${BB_ENCODE_CHUNK_ID}.done"
