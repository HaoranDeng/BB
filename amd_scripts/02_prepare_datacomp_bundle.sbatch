#!/usr/bin/env bash
#SBATCH --job-name=bb-dc-bundle
#SBATCH --partition=mi2101x
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

if [[ -z "${BB_PREPARE_BUNDLE_ID:-}" ]]; then
  echo "BB_PREPARE_BUNDLE_ID is required" >&2
  exit 2
fi
if [[ -z "${BB_PREPARE_BUNDLE_LIST:-}" || ! -s "$BB_PREPARE_BUNDLE_LIST" ]]; then
  echo "BB_PREPARE_BUNDLE_LIST must point to a non-empty file" >&2
  exit 2
fi

bundle_metadata_dir="$BB_MANIFEST_DIR/bundle_${BB_PREPARE_BUNDLE_ID}_metadata"
bundle_output_dir="$BB_WDS_DIR/bundle_${BB_PREPARE_BUNDLE_ID}"

rm -rf "$bundle_metadata_dir"
mkdir -p "$bundle_metadata_dir" "$bundle_output_dir"

idx=0
while IFS= read -r parquet_path; do
  [[ -n "$parquet_path" ]] || continue
  if [[ ! -f "$parquet_path" ]]; then
    echo "Missing parquet file: $parquet_path" >&2
    exit 2
  fi
  ln -s "$parquet_path" "$bundle_metadata_dir/$(printf '%05d' "$idx")_$(basename "$parquet_path")"
  idx=$((idx + 1))
done < "$BB_PREPARE_BUNDLE_LIST"

print_bb_env
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "BB_PREPARE_BUNDLE_ID=$BB_PREPARE_BUNDLE_ID"
echo "BB_PREPARE_BUNDLE_LIST=$BB_PREPARE_BUNDLE_LIST"
echo "bundle_output_dir=$bundle_output_dir"
echo "bundle_file_count=$idx"

"$BB_PYTHON" scripts/prepare_datacomp.py \
  --metadata-dir "$bundle_metadata_dir" \
  --output-dir "$bundle_output_dir" \
  --image-size "$BB_IMAGE_SIZE" \
  --processes "$BB_PREPARE_PROCESSES" \
  --threads "$BB_PREPARE_THREADS"
