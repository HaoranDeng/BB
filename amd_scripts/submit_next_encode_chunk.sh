#!/usr/bin/env bash
set -euo pipefail

cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs

running="$(squeue -h -u "$USER" -n bb-encode-chunk -o '%i' | wc -l)"
if [[ "$running" -gt 0 ]]; then
  echo "encode_chunk_job_already_running=$running"
  exit 0
fi

next=""
for chunk in "$BB_MANIFEST_DIR"/encode_chunk_*.txt; do
  [[ -e "$chunk" ]] || break
  done="${chunk%.txt}.done"
  if [[ ! -e "$done" ]]; then
    next="$chunk"
    break
  fi
done

if [[ -z "$next" ]]; then
  echo "all_encode_chunks_done"
  exit 0
fi

chunk_name="$(basename "$next" .txt)"
chunk_id="${chunk_name#encode_chunk_}"
job_id="$(sbatch --parsable \
  --partition="$BB_GPU_PARTITION" \
  --export=ALL,BB_ENCODE_CHUNK_ID="$chunk_id" \
  amd_scripts/03_encode_manifest_chunk.sbatch)"

echo "submitted_encode_chunk=$chunk_id"
echo "job_id=$job_id"
