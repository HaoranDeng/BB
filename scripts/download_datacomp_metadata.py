from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="mlfoundations/datacomp_1b")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision)
    parquet_files = sorted(path for path in files if fnmatch(path, args.pattern))
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        raise RuntimeError(f"No files matched pattern {args.pattern!r} in {args.repo_id}")

    for path in tqdm(parquet_files, desc="metadata"):
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=path,
            revision=args.revision,
            local_dir=local_dir,
        )

    print(f"downloaded={len(parquet_files)} local_dir={local_dir}")


if __name__ == "__main__":
    main()
