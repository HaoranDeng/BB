# AMD HPC Scripts

These scripts are meant to run on `amdhpc` from `~/workspace/BB`. Heavy data
work is submitted through Slurm with `sbatch`; do not run DataComp download,
image fetching, or feature encoding directly on the login node.

## Defaults

- Repo: `~/workspace/BB`
- Data root: `/work1/jasoncong/denghaoran/BB/data`
- Metadata: `$BB_DATA_ROOT/datacomp_1b/metadata`
- WebDataset shards: `$BB_DATA_ROOT/datacomp_1b/wds_512`
- Encoded latents/text: `$BB_DATA_ROOT/datacomp_1b/encoded_512`
- CPU-only partition: `mi2101x`
- GPU partition: `mi3008x`

Override any setting by exporting it before submission, for example:

```bash
export BB_DATACOMP_MAX_FILES=128
export BB_ENCODE_SHARDS_PER_TASK=4
export BB_PREPARE_ARRAY_CONCURRENCY=8
bash amd_scripts/submit_data_pipeline.sh
```

## Recommended Flow

```bash
cd ~/workspace/BB

# Optional sanity check for the Python environment.
sbatch --partition=mi2101x amd_scripts/00_check_env.sbatch

# If the check reports a missing data-prep package, install the lightweight
# non-Torch dependency bundle through Slurm as well.
sbatch --partition=mi2101x amd_scripts/00_install_deps.sbatch

# Submit the full data preparation chain:
# metadata -> mi210 WebDataset image-download array -> GPU encode array submitter.
bash amd_scripts/submit_data_pipeline.sh

# Watch progress.
bash amd_scripts/status.sh
```

Once `encoded_512` contains `.pt` shards, start the 8x MI300 run:

```bash
sbatch amd_scripts/05_train_8mi300.sbatch
```

This cluster/account currently rejects jobs longer than 4 hours, so long-running
data work is split into arrays and each batch script stays below that limit.

## Python Environment

The scripts use the first available option:

1. `BB_CONDA_ENV`, if set.
2. `$BB_REPO_DIR/.venv`, if it exists.
3. `python` on `PATH`.

For ROCm, prefer a server-side conda environment that already has a matching
ROCm PyTorch build. Then submit like:

```bash
export BB_CONDA_ENV=your_rocm_env
bash amd_scripts/submit_data_pipeline.sh
```
