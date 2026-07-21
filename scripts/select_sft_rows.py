#!/usr/bin/env python3
"""Select deterministic SFT subsets while keeping parent IDs disjoint."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_retry_format(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    messages = row.get("prompt", [])
    if len(messages) != 2 or messages[1].get("role") != "user":
        return row
    content = messages[1].get("content", "")
    match = re.fullmatch(
        r"Fix `solution\.py`.*?\n\nTask:\n(.*?)\n\nCurrent file:\n```python\n(.*?)\n```"
        r"\n\nTests:\n```python\n(.*?)\n```\n\nTest failure:\n```text\n(.*?)\n```",
        content,
        flags=re.DOTALL,
    )
    if not match:
        return row
    task, mutant, tests, failure = match.groups()
    initial = (
        f"Implement `solution.py` for this task. Return the complete file.\n\nTask:\n{task.strip()}"
        f"\n\nTests:\n```python\n{tests.strip()}\n```"
    )
    retry = f"The tests failed:\n```text\n{failure.strip()}\n```\nFix the file and return the complete corrected file."
    row["prompt"] = [
        messages[0],
        {"role": "user", "content": initial},
        {"role": "assistant", "content": f"solution.py\n```python\n{mutant.strip()}\n```"},
        {"role": "user", "content": retry},
    ]
    row["format"] = "retry"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind")
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-validation", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--retry-format", action="store_true")
    args = parser.parse_args()

    train = read_jsonl(Path(args.train_file))
    validation = read_jsonl(Path(args.validation_file))
    if args.kind:
        train = [row for row in train if row.get("kind") == args.kind]
        validation = [row for row in validation if row.get("kind") == args.kind]
    if args.retry_format:
        train = [to_retry_format(row) for row in train]
        validation = [to_retry_format(row) for row in validation]

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    train = train[: args.max_train]
    validation = validation[: args.max_validation]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    metadata = {
        "train": len(train),
        "validation": len(validation),
        "kind": args.kind,
        "seed": args.seed,
        "retry_format": args.retry_format,
        "train_parents": len({row.get("parent_id") for row in train}),
        "validation_parents": len({row.get("parent_id") for row in validation}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
