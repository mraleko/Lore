#!/usr/bin/env python3
"""Create a deterministic mixed SFT split from existing chat JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number} missing non-empty messages")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from only {len(rows)} available")
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    return [rows[index] for index in indexes[:count]]


def add_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [{**row, "source": source} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nextcoder-train", default="data/processed/nextcoder_python_v3_fast/train.jsonl")
    parser.add_argument("--nextcoder-validation", default="data/processed/nextcoder_python_v3_fast/validation.jsonl")
    parser.add_argument("--fable-train", default="data/processed/fable5_2k_fast/train.jsonl")
    parser.add_argument("--fable-validation", default="data/processed/fable5_2k_fast/validation.jsonl")
    parser.add_argument("--output-dir", default="data/processed/mixed_nextcoder_fable_v4")
    parser.add_argument("--total-train", type=int, default=2000)
    parser.add_argument("--total-validation", type=int, default=40)
    parser.add_argument("--fable-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not 0 <= args.fable_ratio <= 1:
        raise ValueError("--fable-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    nextcoder_train = read_jsonl(Path(args.nextcoder_train))
    nextcoder_validation = read_jsonl(Path(args.nextcoder_validation))
    fable_train = read_jsonl(Path(args.fable_train))
    fable_validation = read_jsonl(Path(args.fable_validation))

    train_fable_count = round(args.total_train * args.fable_ratio)
    train_nextcoder_count = args.total_train - train_fable_count
    validation_fable_count = round(args.total_validation * args.fable_ratio)
    validation_nextcoder_count = args.total_validation - validation_fable_count

    train_rows = (
        add_source(sample_rows(nextcoder_train, train_nextcoder_count, rng), "nextcoder_python_v3_fast")
        + add_source(sample_rows(fable_train, train_fable_count, rng), "fable5_2k_fast")
    )
    validation_rows = (
        add_source(sample_rows(nextcoder_validation, validation_nextcoder_count, rng), "nextcoder_python_v3_fast")
        + add_source(sample_rows(fable_validation, validation_fable_count, rng), "fable5_2k_fast")
    )
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)

    metadata = {
        "seed": args.seed,
        "fable_ratio": args.fable_ratio,
        "train": {
            "total": len(train_rows),
            "nextcoder_python_v3_fast": train_nextcoder_count,
            "fable5_2k_fast": train_fable_count,
        },
        "validation": {
            "total": len(validation_rows),
            "nextcoder_python_v3_fast": validation_nextcoder_count,
            "fable5_2k_fast": validation_fable_count,
        },
        "inputs": {
            "nextcoder_train": args.nextcoder_train,
            "nextcoder_validation": args.nextcoder_validation,
            "fable_train": args.fable_train,
            "fable_validation": args.fable_validation,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
