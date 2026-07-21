#!/usr/bin/env python3
"""Build token-safe, test-verified Python solution and repair SFT data."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM = (
    "You are editing a Python file. Return the complete corrected file using the whole-file format: "
    "the filename on one line followed by a fenced python code block. Do not omit unchanged code."
)
SAFE_LIBS = {
    "array", "base64", "bisect", "calendar", "collections", "copy", "csv", "datetime", "decimal",
    "difflib", "enum", "fractions", "functools", "hashlib", "heapq", "html", "io", "itertools",
    "json", "math", "operator", "pathlib", "random", "re", "statistics", "string", "textwrap",
    "time", "typing", "unittest", "urllib", "uuid",
}
UNSAFE_TEXT = ("subprocess", "socket", "requests", "urlopen", "http.client", "os.system", "shutil.rmtree")


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def code_block(code: str) -> str:
    return f"solution.py\n```python\n{code.rstrip()}\n```"


def normalize_failure(output: str) -> str:
    output = re.sub(r'File "[^"]+"', 'File "solution.py"', output)
    output = re.sub(r"0x[0-9a-fA-F]+", "0x...", output)
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-35:])[-3500:]


def run_tests(code: str, tests: str, timeout: int) -> tuple[bool, str]:
    script = code.rstrip() + "\n\n" + tests.rstrip() + "\n"
    if "unittest" in tests and "unittest.main(" not in tests:
        script += "\nif __name__ == '__main__':\n    unittest.main()\n"
    try:
        ast.parse(script)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    with tempfile.TemporaryDirectory(prefix="v5-verify-") as directory:
        path = Path(directory) / "solution.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return False, "TEST TIMEOUT"
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, normalize_failure(output)


def mutate_candidates(code: str, seed: int) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    candidates: list[tuple[str, int]] = []
    for index, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            candidates.append(("comparison", index))
        elif isinstance(node, ast.BoolOp):
            candidates.append(("boolean", index))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            candidates.append(("arithmetic", index))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            candidates.append(("state_update", index))
        elif isinstance(node, (ast.If, ast.While)):
            candidates.append(("guard", index))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            candidates.append(("range_boundary", index))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    results: list[tuple[str, str]] = []
    for kind, target_index in candidates[:30]:
        mutant = copy.deepcopy(tree)
        nodes = list(ast.walk(mutant))
        if target_index >= len(nodes):
            continue
        node = nodes[target_index]
        if kind == "comparison":
            mapping = {
                ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
                ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
                ast.Is: ast.IsNot, ast.IsNot: ast.Is,
            }
            replacement = mapping.get(type(node.ops[0]))
            if not replacement:
                continue
            node.ops[0] = replacement()
        elif kind == "boolean":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif kind == "arithmetic":
            mapping = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
            node.op = mapping[type(node.op)]()
        elif kind == "state_update":
            node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
        elif kind == "guard":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif kind == "range_boundary":
            arg_index = len(node.args) - 1
            node.args[arg_index] = ast.BinOp(left=node.args[arg_index], op=ast.Add(), right=ast.Constant(value=1))
        ast.fix_missing_locations(mutant)
        mutated = ast.unparse(mutant)
        if mutated != code:
            results.append((kind, mutated))
    return results


def make_solution_row(source: str, parent_id: str, description: str, tests: str, solution: str) -> dict[str, Any]:
    user = f"Implement `solution.py` for this task.\n\nTask:\n{description.strip()}\n\nTests:\n```python\n{tests.strip()}\n```"
    return {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": code_block(solution)}],
        "source": source,
        "parent_id": parent_id,
        "kind": "solution",
    }


def make_repair_row(
    source: str, parent_id: str, description: str, tests: str, mutant: str, failure: str,
    solution: str, mutation: str,
) -> dict[str, Any]:
    user = (
        f"Fix `solution.py` so it satisfies the task and tests. Return the complete file.\n\nTask:\n{description.strip()}"
        f"\n\nCurrent file:\n```python\n{mutant.rstrip()}\n```\n\nTests:\n```python\n{tests.strip()}\n```"
        f"\n\nTest failure:\n```text\n{failure.strip()}\n```"
    )
    return {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": code_block(solution)}],
        "source": source,
        "parent_id": parent_id,
        "kind": "repair",
        "mutation": mutation,
    }


def token_safe(row: dict[str, Any], tokenizer: Any, prompt_limit: int, completion_limit: int, total_limit: int) -> bool:
    prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    completion_text = tokenizer.apply_chat_template(row["completion"], tokenize=False, add_generation_prompt=False)
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(completion_text, add_special_tokens=False))
    row["prompt_tokens"] = prompt_tokens
    row["completion_tokens"] = completion_tokens
    return prompt_tokens <= prompt_limit and completion_tokens <= completion_limit and prompt_tokens + completion_tokens <= total_limit


def verified_rows(
    source: str, parent_id: str, description: str, tests: str, solution: str, tokenizer: Any,
    args: argparse.Namespace, skipped: Counter[str],
) -> list[dict[str, Any]]:
    if any(text in (solution + tests).lower() for text in UNSAFE_TEXT):
        skipped["unsafe"] += 1
        return []
    passed, output = run_tests(solution, tests, args.test_timeout)
    if not passed:
        skipped["reference_failed"] += 1
        return []
    rows = [make_solution_row(source, parent_id, description, tests, solution)]
    repair_count = 0
    for mutation, mutant in mutate_candidates(solution, stable_int(f"{args.seed}:{source}:{parent_id}")):
        mutant_passed, failure = run_tests(mutant, tests, args.test_timeout)
        if mutant_passed or failure == "TEST TIMEOUT":
            continue
        rows.append(make_repair_row(source, parent_id, description, tests, mutant, failure, solution, mutation))
        repair_count += 1
        if repair_count >= args.repairs_per_parent:
            break
    accepted = []
    for row in rows:
        if token_safe(row, tokenizer, args.max_prompt_tokens, args.max_completion_tokens, args.max_total_tokens):
            accepted.append(row)
        else:
            skipped["too_long"] += 1
    if len(accepted) == 1:
        skipped["no_verified_mutant"] += 1
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed/v5_verified_repair")
    parser.add_argument("--tokenizer", default="models/qwen25-coder-7b-mixed-nextcoder-fable-v4-lora")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-examples", type=int, default=2200)
    parser.add_argument("--max-bigcode-parents", type=int, default=700)
    parser.add_argument("--repairs-per-parent", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=700)
    parser.add_argument("--max-completion-tokens", type=int, default=300)
    parser.add_argument("--max-total-tokens", type=int, default=1000)
    parser.add_argument("--test-timeout", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    skipped: Counter[str] = Counter()
    parents: list[tuple[str, str, list[dict[str, Any]]]] = []

    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    for row in mbpp:
        parent_id = f"mbpp-{row['task_id']}"
        tests = "\n".join(row["test_list"])
        rows = verified_rows("mbpp", parent_id, row["text"], tests, row["code"], tokenizer, args, skipped)
        if rows:
            parents.append(("mbpp", parent_id, rows))

    quix = load_dataset("Muennighoff/quixbugs", split="train")
    for row in quix:
        parent_id = f"quixbugs-{row['name']}"
        rows = verified_rows(
            "quixbugs", parent_id, row["docstring"], row["tests"], row["solution"], tokenizer, args, skipped,
        )
        if rows:
            repair = make_repair_row(
                "quixbugs", parent_id, row["docstring"], row["tests"], row["buggy_program"],
                run_tests(row["buggy_program"], row["tests"], args.test_timeout)[1], row["solution"], "upstream_bug",
            )
            if token_safe(repair, tokenizer, args.max_prompt_tokens, args.max_completion_tokens, args.max_total_tokens):
                rows = [repair]
            parents.append(("quixbugs", parent_id, rows))

    bigcode = load_dataset("bigcode/bigcodebench", split="v0.1.4", streaming=True)
    bigcode_kept = 0
    for row in bigcode:
        if bigcode_kept >= args.max_bigcode_parents:
            break
        try:
            libs = set(ast.literal_eval(row.get("libs", "[]")))
        except (ValueError, SyntaxError):
            skipped["bad_libs"] += 1
            continue
        if not libs.issubset(SAFE_LIBS):
            skipped["external_libs"] += 1
            continue
        solution = row["code_prompt"].rstrip() + "\n" + row["canonical_solution"].lstrip("\n")
        parent_id = row["task_id"]
        rows = verified_rows(
            "bigcodebench", parent_id, row["instruct_prompt"], row["test"], solution, tokenizer, args, skipped,
        )
        if rows:
            parents.append(("bigcodebench", parent_id, rows))
            bigcode_kept += 1

    rng = random.Random(args.seed)
    rng.shuffle(parents)
    validation_parent_ids = {parent_id for _, parent_id, _ in parents[: max(20, len(parents) // 50)]}
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for _, parent_id, rows in parents:
        target = validation_rows if parent_id in validation_parent_ids else train_rows
        target.extend(rows)
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    train_rows = train_rows[: args.max_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "train": len(train_rows),
        "validation": len(validation_rows),
        "train_sources": dict(Counter(row["source"] for row in train_rows)),
        "train_kinds": dict(Counter(row["kind"] for row in train_rows)),
        "validation_parents": len(validation_parent_ids),
        "skipped": dict(skipped),
        "args": vars(args),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
