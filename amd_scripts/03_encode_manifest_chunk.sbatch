#!/usr/bin/env bash
#SBATCH --job-name=bb-encode-chunk
#SBATCH --partition=mi3008x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

"$BB_PYTHON" scripts/encode_shards.py \
  --input-shards "@$chunk_list" \
  --output-dir "$BB_ENCODED_DIR" \
  --output-prefix "chunk_${BB_ENCODE_CHUNK_ID}_" \
  --image-size "$BB_IMAGE_SIZE" \
  --batch-size "$BB_ENCODE_BATCH_SIZE" \
  --workers "$BB_ENCODE_WORKERS" \
  --samples-per-shard "$BB_ENCODE_SAMPLES_PER_SHARD" \
  --precision bf16

touch "$BB_MANIFEST_DIR/encode_chunk_${BB_ENCODE_CHUNK_ID}.done"
