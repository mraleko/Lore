#!/usr/bin/env python3
"""Convert Fable 5-style Hugging Face data into chat JSONL for SFT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
}


def normalize_role(role: Any) -> str:
    return ROLE_MAP.get(str(role).lower(), str(role).lower())


def get_content(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = normalize_role(message.get("role", message.get("from", message.get("speaker", ""))))
    content = get_content(message)
    has_tool_calls = bool(message.get("tool_calls"))
    if not role or (not content and not has_tool_calls):
        return None

    normalized: dict[str, Any] = {"role": role, "content": content}
    for key in ("tool_calls", "tool_call_id", "name"):
        if key in message:
            normalized[key] = message[key]
    return normalized


def normalize_messages(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if not isinstance(value, list):
        return None

    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        message = normalize_message(item)
        if message:
            messages.append(message)

    has_user = any(message["role"] == "user" for message in messages)
    has_assistant = any(message["role"] == "assistant" for message in messages)
    if has_user and has_assistant:
        return messages
    return None


def row_to_messages(row: dict[str, Any], text_column: str | None = None) -> list[dict[str, Any]] | None:
    if text_column:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            return [{"role": "user", "content": text.strip()}, {"role": "assistant", "content": ""}]

    for key in ("messages", "conversations", "conversation", "chat", "turns"):
        messages = normalize_messages(row.get(key))
        if messages:
            return messages

    prompt = None
    response = None
    for key in ("prompt", "instruction", "input", "question", "task"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break
    for key in ("response", "output", "answer", "completion", "assistant"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            response = value.strip()
            break

    if prompt and response:
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Nexlab/fable5-agentic-coding-sft")
    parser.add_argument("--config", help="Optional dataset config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/processed/fable5")
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--text-column", help="Fallback single text column; prefer structured columns when possible")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    examples: list[dict[str, Any]] = []
    skipped = 0

    for row in ds:
        messages = row_to_messages(row, args.text_column)
        if not messages:
            skipped += 1
            continue
        examples.append({"messages": messages})
        if len(examples) >= args.max_examples:
            break

    if not examples:
        raise SystemExit("No usable chat examples found. Inspect the dataset and add column handling.")

    random.Random(args.seed).shuffle(examples)
    validation_size = max(1, int(len(examples) * args.validation_ratio)) if len(examples) > 1 else 0
    validation = examples[:validation_size]
    train = examples[validation_size:]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    if validation:
        write_jsonl(output_dir / "validation.jsonl", validation)

    print(f"Wrote {len(train)} train examples")
    print(f"Wrote {len(validation)} validation examples")
    print(f"Skipped {skipped} rows")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
