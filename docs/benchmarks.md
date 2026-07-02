# Benchmark Plan

The goal is to quantify whether Fable 5 agentic-coding fine-tuning improves a local Qwen2.5-Coder 7B model.

## Primary Benchmark: aider

Use aider as the main benchmark because it measures repository-editing behavior rather than isolated coding snippets.

External reference baseline from aider's old code-editing leaderboard:

| Model | Score | Edit Format Score | Command | Edit Format |
| --- | ---: | ---: | --- | --- |
| `qwen2.5-coder:7b-instruct-q8_0` | 51.9% | 100.0% | `aider --model ollama/qwen2.5-coder:7b-instruct-q8_0` | whole |

This is useful as a sanity-check target, but it should not be treated as the actual baseline for this project. The leaderboard page says this old code-editing benchmark has been replaced by the newer polyglot leaderboard, and your local setup may differ by aider version, Ollama version, quantization, prompt/model settings, hardware, and benchmark runner version.

Recommended flow:

1. Run the benchmark with the base model.
2. Train the LoRA adapter.
3. Run the same benchmark with the fine-tuned model.
4. Compare solve rate, edit validity, test pass rate, and regression patterns.

Set up local benchmark repos:

```bash
bash scripts/setup_aider_benchmark.sh
```

Start with a 10-task baseline smoke run inside aider's Docker container:

```bash
export OLLAMA_API_BASE=http://host.docker.internal:11434

./benchmark/benchmark.py qwen25-coder-7b-q8-local-baseline \
  --model ollama_chat/qwen2.5-coder:7b-instruct-q8_0 \
  --edit-format whole \
  --threads 1 \
  --num-tests 10 \
  --exercises-dir polyglot-benchmark
```

Then run the full local baseline by removing `--num-tests 10`.

The old leaderboard command used `ollama/qwen2.5-coder:7b-instruct-q8_0`; current aider Ollama docs recommend `ollama_chat/qwen2.5-coder:7b-instruct-q8_0`. Use one consistently across baseline and fine-tuned runs.

First local smoke result:

| Run | Test Cases | Model | Pass Rate 1 | Pass Rate 2 | Well-Formed | Seconds/Case | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `2026-07-02-01-41-06--qwen25-coder-7b-q8-smoke-1` | 1 | `ollama_chat/qwen2.5-coder:7b-instruct-q8_0` | 0.0% | 0.0% | 100.0% | 146.2 | Harness works; one-task smoke failed on `rest-api` |

Small matched `rest-api` comparison:

| Run | Test Cases | Model | Pass Rate 1 | Pass Rate 2 | Well-Formed | User Asks | Seconds/Case | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `2026-07-02-05-38-14--qwen25-coder-7b-q8-rest-api` | 3 | `ollama_chat/qwen2.5-coder:7b-instruct-q8_0` | 0.0% | 0.0% | 100.0% | 2 | 160.3 | Base Q8 model |
| `2026-07-02-05-27-21--qwen25-coder-7b-fable5-2k-fast-rest-api` | 3 | `ollama_chat/qwen25-coder-7b-fable5-2k-fast-q8` | 0.0% | 0.0% | 100.0% | 0 | 144.9 | First Fable 5 LoRA export |

This matched smoke set is too small to judge benchmark improvement. It confirms the exported model is benchmarkable, but a larger sample is needed.

Keep benchmark data completely out of the training set. Do not train on aider benchmark tasks if the score is meant to represent real improvement.

## Secondary Benchmarks

Use these as quick sanity checks:

- HumanEval+
- MBPP+
- BigCodeBench

These are easier to run and compare, but they are less aligned with aider because they do not fully test multi-file edit behavior.

## Expected Failure Modes

- Overfitting to trace style instead of solving tasks.
- Excessive tool-call or narration behavior when plain code is needed.
- Longer, riskier patches that pass fewer tests.
- Lower benchmark scores despite better-looking conversations.

## Practical Tracking

Create a simple table for each run:

| Run | Model | Dataset | Examples | Seq Len | LoRA Rank | Benchmark | Score | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| external-reference | qwen2.5-coder:7b-instruct-q8_0 | none | 0 | n/a | n/a | aider old edit benchmark | 51.9% | public leaderboard reference, not local baseline |
| local-baseline | Qwen2.5-Coder-7B-Instruct | none | 0 | n/a | n/a | aider | TBD | run locally before training |
| fable5-sft-v1 | Qwen2.5-Coder-7B + LoRA | Fable 5 | TBD | 4096 | 16 | aider | TBD | first pass |

Training note: the first `fable5_2k` run with `max_seq_length=4096` reached step `156/245` before the 2-hour command timeout and only saved `checkpoint-100`. Some long tool-trace examples made step times impractical.

Completed training run:

| Run | Dataset | Train | Validation | Seq Len | Steps | Runtime | Train Loss | Eval Loss | Output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fable5-2k-fast-lora` | `fable5_2k_fast` | 1960 | 40 | 2048 | 245 | 2711s | 1.373 | 1.384 | `models/qwen25-coder-7b-fable5-2k-fast-lora` |
