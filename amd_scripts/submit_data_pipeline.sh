#!/usr/bin/env bash
set -euo pipefail

cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs

meta_job="$(sbatch --parsable \
  --partition="$BB_CPU_PARTITION" \
  --export=ALL \
  amd_scripts/01_download_metadata.sbatch)"

prepare_job="$(sbatch --parsable \
  --partition="$BB_CPU_PARTITION" \
  --dependency="afterok:$meta_job" \
  --export=ALL \
  amd_scripts/02_prepare_datacomp.sbatch)"

encode_submit_job="$(sbatch --parsable \
  --partition="$BB_DEVEL_PARTITION" \
  --dependency="afterok:$prepare_job" \
  --export=ALL \
  amd_scripts/04_submit_encode_array.sbatch)"

cat <<EOF
Submitted DataComp pipeline:
  metadata job:       $meta_job
  webdataset job:     $prepare_job
  encode submit job:  $encode_submit_job

The encode submit job will create shard-list manifests and submit the actual GPU
array after the WebDataset job succeeds. Watch with:
  squeue -u "$USER"
  tail -f "$BB_LOG_DIR"/bb-dc-meta-${meta_job}.out
EOF
