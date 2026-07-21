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

BLOCKED_SUBSTRINGS = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "api key:",
    "api_key",
    "gsk_",
    "sk-",
    "hf_",
)
CODE_EDIT_KEYWORDS = (
    "bug",
    "debug",
    "error",
    "exception",
    "fail",
    "fix",
    "implement",
    "refactor",
    "test",
    "function",
    "class",
    "method",
    "script",
    "code",
    "python",
    "javascript",
    "typescript",
    "react",
    "node",
    "api",
)
BROAD_APP_KEYWORDS = (
    "build me an app",
    "create an app",
    "make me an app",
    "build a game",
    "create a game",
    "make me a game",
    "build me a game",
    "full app",
    "full-stack app",
    "entire website",
    "make me a website",
    "clone of",
    "similar to rainbow six",
)
DEFAULT_DATA_FILES = {
    "Nexlab/fable5-agentic-coding-sft": "https://huggingface.co/datasets/Nexlab/fable5-agentic-coding-sft/resolve/main/sft_curated_full.jsonl",
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
        if message.get(key) is not None:
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


def has_blocked_content(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = str(message.get("content", "")).lower()
        if any(blocked in content for blocked in BLOCKED_SUBSTRINGS):
            return True
        for tool_call in message.get("tool_calls", []) or []:
            text = json.dumps(tool_call, ensure_ascii=False).lower()
            if any(blocked in text for blocked in BLOCKED_SUBSTRINGS):
                return True
    return False


def message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(str(message.get("content", "")))
        for tool_call in message.get("tool_calls", []) or []:
            total += len(json.dumps(tool_call, ensure_ascii=False))
    return total


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(message.get("tool_calls", []) or [])
        if message.get("role") == "tool":
            total += 1
    return total


def combined_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if content:
            parts.append(str(content))
    return "\n".join(parts).lower()


def first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).lower()
    return ""


def assistant_text_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages if message.get("role") == "assistant")


def has_code_edit_signal(messages: list[dict[str, Any]]) -> bool:
    text = combined_text(messages)
    return any(keyword in text for keyword in CODE_EDIT_KEYWORDS)


def is_broad_generation_request(messages: list[dict[str, Any]]) -> bool:
    text = first_user_text(messages)
    return any(keyword in text for keyword in BROAD_APP_KEYWORDS)


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
    parser.add_argument("--data-file", help="Optional JSONL path or URL to bypass dataset metadata")
    parser.add_argument("--output-dir", default="data/processed/fable5")
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--text-column", help="Fallback single text column; prefer structured columns when possible")
    parser.add_argument("--max-chars", type=int, default=24000, help="Skip examples with more approximate message characters")
    parser.add_argument("--code-edit-clean", action="store_true", help="Keep a stricter code-edit/debug subset")
    parser.add_argument("--max-tool-calls", type=int, help="Skip examples with more tool calls/tool messages")
    parser.add_argument("--max-assistant-chars", type=int, help="Skip examples with more assistant text characters")
    args = parser.parse_args()

    data_file = args.data_file or DEFAULT_DATA_FILES.get(args.dataset)
    if data_file:
        ds = load_dataset("json", data_files={args.split: data_file}, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    examples: list[dict[str, Any]] = []
    skipped = 0

    for row in ds:
        messages = row_to_messages(row, args.text_column)
        if not messages:
            skipped += 1
            continue
        if has_blocked_content(messages):
            skipped += 1
            continue
        if args.max_chars and message_chars(messages) > args.max_chars:
            skipped += 1
            continue
        if args.max_tool_calls is not None and count_tool_calls(messages) > args.max_tool_calls:
            skipped += 1
            continue
        if args.max_assistant_chars is not None and assistant_text_chars(messages) > args.max_assistant_chars:
            skipped += 1
            continue
        if args.code_edit_clean and (not has_code_edit_signal(messages) or is_broad_generation_request(messages)):
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
