from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Report best ViT validation metrics.")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    args = parser.parse_args()

    records = load_records(args.metrics)
    val_records = [item for item in records if item.get("split") == "val"]
    if not val_records:
        raise SystemExit(f"No validation records found in {args.metrics}")

    best = max(val_records, key=lambda item: float(item["acc1"]))
    latest = val_records[-1]
    print(
        "best "
        f"epoch={best['epoch']} step={best['step']} "
        f"loss={best['loss']:.4f} acc1={best['acc1']:.2f} acc5={best['acc5']:.2f}"
    )
    print(
        "latest "
        f"epoch={latest['epoch']} step={latest['step']} "
        f"loss={latest['loss']:.4f} acc1={latest['acc1']:.2f} acc5={latest['acc5']:.2f}"
    )


if __name__ == "__main__":
    main()
