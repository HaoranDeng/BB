from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def parquet_is_readable(path: Path) -> bool:
    try:
        pq.ParquetFile(path)
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wds-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    wds_dir = Path(args.wds_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for tar_path in sorted(wds_dir.rglob("*.tar")):
        parquet_path = tar_path.with_suffix(".parquet")
        if parquet_path.exists() and parquet_is_readable(parquet_path):
            shards.append(tar_path)

    if not shards:
        raise SystemExit(f"No valid WebDataset shards found under {wds_dir}")

    for old in manifest_dir.glob("encode_chunk_*.txt"):
        old.unlink()

    chunks = 0
    for index in range(0, len(shards), args.chunk_size):
        chunk = shards[index : index + args.chunk_size]
        out = manifest_dir / f"encode_chunk_{chunks:05d}.txt"
        out.write_text("\n".join(str(path) for path in chunk) + "\n", encoding="utf-8")
        chunks += 1

    summary = manifest_dir / "encode_chunks.summary"
    summary.write_text(
        f"valid_shards={len(shards)}\nchunk_size={args.chunk_size}\nchunks={chunks}\n",
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
