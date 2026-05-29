#!/usr/bin/env bash
set -euo pipefail

cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs

meta_job="$(sbatch --parsable \
  --partition="$BB_CPU_PARTITION" \
  --export=ALL \
  amd_scripts/01_download_metadata.sbatch)"

prepare_submit_job="$(sbatch --parsable \
  --partition="$BB_CPU_PARTITION" \
  --dependency="afterok:$meta_job" \
  --export=ALL \
  amd_scripts/02_submit_prepare_array.sbatch)"

cat <<EOF
Submitted DataComp pipeline:
  metadata job:         $meta_job
  prepare submit job:   $prepare_submit_job

The prepare submit job will create a WebDataset download array on mi210 and then
queue the encode-array submitter after the download array succeeds. Watch with:
  squeue -u "$USER"
  tail -f "$BB_LOG_DIR"/bb-dc-meta-${meta_job}.out
EOF
