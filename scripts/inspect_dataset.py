#!/usr/bin/env python3
"""Inspect a Hugging Face dataset schema and a few rows."""

from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import get_dataset_config_names, load_dataset


DEFAULT_DATA_FILES = {
    "Nexlab/fable5-agentic-coding-sft": "https://huggingface.co/datasets/Nexlab/fable5-agentic-coding-sft/resolve/main/sft_curated_full.jsonl",
}


def compact(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... <truncated>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Hugging Face dataset id")
    parser.add_argument("--config", help="Optional dataset config name")
    parser.add_argument("--split", default="train", help="Split to inspect")
    parser.add_argument("--data-file", help="Optional JSONL path or URL to bypass dataset metadata")
    parser.add_argument("--rows", type=int, default=3, help="Rows to print")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max chars per row")
    args = parser.parse_args()

    data_file = args.data_file or DEFAULT_DATA_FILES.get(args.dataset)

    if args.config is None and not data_file:
        try:
            configs = get_dataset_config_names(args.dataset)
        except Exception as exc:  # noqa: BLE001
            configs = []
            print(f"Could not list configs: {exc}")
        if configs:
            print("Configs:")
            for name in configs[:50]:
                print(f"- {name}")
            if len(configs) > 1:
                print("\nMultiple configs found. Re-run with --config if needed.\n")

    if data_file:
        ds = load_dataset("json", data_files={args.split: data_file}, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config or '<default>'}")
    print(f"Split: {args.split}")

    for index, row in enumerate(ds.take(args.rows), start=1):
        print(f"\n--- row {index} ---")
        print("Columns:", ", ".join(row.keys()))
        print(compact(row, args.max_chars))


if __name__ == "__main__":
    main()
