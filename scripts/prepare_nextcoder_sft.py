#!/usr/bin/env python3
"""Prepare filtered NextCoder code-edit examples as chat JSONL for SFT."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


PYTHON_SIGNALS = (
    "```python",
    "```py",
    "def ",
    "class ",
    "import ",
    "from ",
    "pytest",
    "unittest",
    ".py",
    "python",
)
NON_PYTHON_FENCES = (
    "```c",
    "```cpp",
    "```c++",
    "```java",
    "```javascript",
    "```js",
    "```typescript",
    "```ts",
    "```go",
    "```rust",
    "```kotlin",
)
BAD_SUBSTRINGS = (
    "<think>",
    "assistant (tool call)",
    "tool result",
    "request interrupted",
    "api key",
    "sk-",
    "hf_",
)
WHOLE_APP_PHRASES = (
    "build a full",
    "build an app",
    "create an app",
    "make an app",
    "build a game",
    "create a game",
    "clone csgo",
    "entire website",
)


def looks_python(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in PYTHON_SIGNALS)


def has_non_python_fence(text: str) -> bool:
    lower = text.lower()
    return any(fence in lower for fence in NON_PYTHON_FENCES)


def has_bad_content(text: str) -> bool:
    lower = text.lower()
    return any(bad in lower for bad in BAD_SUBSTRINGS)


def is_whole_app_prompt(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in WHOLE_APP_PHRASES)


def strip_outer_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def clean_completion(text: str, strip_fences: bool) -> str:
    text = text.strip()
    if strip_fences:
        text = strip_outer_code_fence(text)
    return text


def row_to_messages(row: dict[str, Any], strip_fences: bool) -> tuple[list[dict[str, str]] | None, str]:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        return None, "missing_fields"

    prompt = prompt.strip()
    completion = clean_completion(completion, strip_fences=strip_fences)
    if not prompt or not completion:
        return None, "empty"

    combined = f"{prompt}\n{completion}"
    if has_bad_content(combined):
        return None, "bad_content"
    if is_whole_app_prompt(prompt):
        return None, "whole_app"
    if has_non_python_fence(prompt) or has_non_python_fence(completion):
        return None, "non_python_fence"
    if not looks_python(combined):
        return None, "not_python"

    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], "kept"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="microsoft/NextCoderDataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/processed/nextcoder_python_v3")
    parser.add_argument("--max-examples", type=int, default=3000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--max-completion-chars", type=int, default=10000)
    parser.add_argument("--min-completion-chars", type=int, default=80)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-fences", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen = 0

    for row in dataset:
        seen += 1
        messages, reason = row_to_messages(row, strip_fences=args.strip_fences)
        if not messages:
            skipped[reason] += 1
            continue

        prompt_len = len(messages[0]["content"])
        completion_len = len(messages[1]["content"])
        if prompt_len > args.max_prompt_chars:
            skipped["prompt_too_long"] += 1
            continue
        if completion_len > args.max_completion_chars:
            skipped["completion_too_long"] += 1
            continue
        if completion_len < args.min_completion_chars:
            skipped["completion_too_short"] += 1
            continue

        rows.append({"messages": messages})
        if len(rows) >= args.max_examples:
            break

    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * args.validation_ratio)) if rows else 0
    validation_rows = rows[:val_size]
    train_rows = rows[val_size:]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "seen": seen,
        "kept": len(rows),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "skipped": dict(skipped),
        "args": vars(args),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
