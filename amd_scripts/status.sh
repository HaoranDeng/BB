#!/usr/bin/env bash
set -euo pipefail

cd "${BB_REPO_DIR:-$HOME/workspace/BB}"
source amd_scripts/common.sh
prepare_bb_dirs

echo "== Slurm queue =="
squeue -u "$USER" || true

echo
echo "== Data counts =="
printf 'metadata parquet: '; find "$BB_METADATA_DIR" -type f -name '*.parquet' 2>/dev/null | wc -l
printf 'webdataset tar:   '; find "$BB_WDS_DIR" -type f -name '*.tar' 2>/dev/null | wc -l
printf 'encoded pt:       '; find "$BB_ENCODED_DIR" -type f -name '*.pt' 2>/dev/null | wc -l

echo
echo "== Latest logs =="
find "$BB_LOG_DIR" -type f \( -name '*.out' -o -name '*.err' \) -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
  2>/dev/null | sort | tail -20
