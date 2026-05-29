from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


URL_COLUMNS = ("url", "URL", "image_url", "image_url_https")
CAPTION_COLUMNS = ("text", "TEXT", "caption", "alt_text")


def infer_column(schema_names: list[str], candidates: tuple[str, ...], label: str) -> str:
    lowered = {name.lower(): name for name in schema_names}
    for candidate in candidates:
        if candidate in schema_names:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Could not infer {label} column from: {schema_names}")


def infer_columns(metadata_dir: Path) -> tuple[str, str]:
    first = next(metadata_dir.rglob("*.parquet"), None)
    if first is None:
        raise FileNotFoundError(f"No parquet files under {metadata_dir}")
    schema = pq.read_schema(first)
    names = list(schema.names)
    return (
        infer_column(names, URL_COLUMNS, "URL"),
        infer_column(names, CAPTION_COLUMNS, "caption"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url-col", default=None)
    parser.add_argument("--caption-col", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--resize-mode", default="center_crop")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inferred_url_col, inferred_caption_col = infer_columns(metadata_dir)
    url_col = args.url_col or inferred_url_col
    caption_col = args.caption_col or inferred_caption_col

    img2dataset_bin = shutil.which("img2dataset")
    if img2dataset_bin is None:
        raise RuntimeError(
            "img2dataset executable not found on PATH. Install it with "
            "`python -m pip install img2dataset`."
        )

    cmd = [
        img2dataset_bin,
        "--url_list",
        str(metadata_dir),
        "--input_format",
        "parquet",
        "--url_col",
        url_col,
        "--caption_col",
        caption_col,
        "--output_format",
        "webdataset",
        "--output_folder",
        str(output_dir),
        "--image_size",
        str(args.image_size),
        "--resize_mode",
        args.resize_mode,
        "--processes_count",
        str(args.processes),
        "--thread_count",
        str(args.threads),
        "--retries",
        str(args.retries),
        "--enable_wandb",
        "False",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
