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

## Old aider Edit Benchmark

For leaderboard-comparable local runs, use the old aider `v0.56.0` harness in `benchmarks/aider-v0.56.0` and the pinned Exercism Python corpus in `benchmarks/aider-v0.56.0/tmp.benchmarks/exercism-python`. This corpus has exactly 133 tasks, matching the public old leaderboard row for `qwen2.5-coder:7b-instruct-q8_0`.

The old Docker image needs a scoped build workaround because the build context does not include `.git` metadata for `setuptools_scm`:

```dockerfile
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT=0.56.0
```

The benchmark checkout is a git worktree, so Docker runs also need the parent aider `.git` metadata mounted at the same absolute path used by the worktree `.git` pointer:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD":/aider \
  -v "$PWD/tmp.benchmarks":/benchmarks \
  -v "/mnt/c/Users/MESHLICIOUS/Downloads/Fine Tune LLM/benchmarks/aider/.git":"/mnt/c/Users/MESHLICIOUS/Downloads/Fine Tune LLM/benchmarks/aider/.git":ro \
  -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434 \
  aider-benchmark \
  ./benchmark/benchmark.py RUN_NAME \
    --model ollama/MODEL_NAME \
    --edit-format whole \
    --threads 1 \
    --num-tests 1 \
    --new
```

One-case old-harness smoke results:

| Run | Test Cases | Model | Pass Rate 1 | Pass Rate 2 | Well-Formed | User Asks | Seconds/Case | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `2026-07-08-23-36-16--smoke-base-qwen25-7b-q8` | 1 | `ollama/qwen2.5-coder:7b-instruct-q8_0` | 0.0% | 0.0% | 100.0% | 0 | 59.1 | Old `v0.56.0` harness works; failed `word-count` |
| `2026-07-08-23-38-12--smoke-ft-qwen25-7b-fable5-q8` | 1 | `ollama/qwen25-coder-7b-fable5-2k-fast-q8` | 0.0% | 0.0% | 100.0% | 0 | 75.5 | Old `v0.56.0` harness works; failed `crypto-square` |
| `2026-07-09-19-52-31--smoke-ft-v2-qwen25-fable5-5k-q8` | 1 | `ollama/qwen25-coder-7b-fable5-5k-fast-q8` | 0.0% | 0.0% | 100.0% | 0 | 17.4 | Old `v0.56.0` harness works; failed `acronym` |
| `2026-07-10-03-46-09--smoke-ft-v15-qwen25-fable5-3k-clean-q8` | 1 | `ollama/qwen25-coder-7b-fable5-3k-clean-q8` | 0.0% | 0.0% | 100.0% | 0 | 83.6 | Old `v0.56.0` harness works; failed `paasio` |
| `2026-07-12-00-05-30--smoke-ft-v4-mixed-nextcoder-fable-q8` | 1 | `ollama/qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 100.0% | n/a | 100.0% | 0 | 25.4 | Old `v0.56.0` harness works; passed `matching-brackets` |

These smoke runs are setup checks only. They are not comparable model scores because each run used one randomly selected task.

Full old-harness comparable result:

| Run | Test Cases | Model | Pass Rate 1 | Pass Rate 2 | Well-Formed | User Asks | Test Timeouts | Seconds/Case | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| public leaderboard reference | 133 | `ollama/qwen2.5-coder:7b-instruct-q8_0` | 45.1% | 51.9% | 100.0% | n/a | n/a | n/a | aider `0.56.0`, `whole` edit format |
| `2026-07-10-21-09-35--full-base-qwen25-coder-7b-instruct-q8` | 133 | `ollama/qwen2.5-coder:7b-instruct-q8_0` | 46.6% | 53.4% | 100.0% | 3 | 2 | 93.8 | Local base model, old aider `0.56.0`, `whole`; run resumed after timeout at 115/133 |
| `2026-07-08-23-43-49--full-ft-qwen25-fable5-2k-fast-q8` | 133 | `ollama/qwen25-coder-7b-fable5-2k-fast-q8` | 49.6% | 54.1% | 100.0% | 11 | 3 | 51.0 | Local fine-tuned model, old aider `0.56.0`, `whole`; `commit_hash` was `6f2b064-dirty` |
| `2026-07-09-21-44-21--full-ft-v1-rerun-qwen25-fable5-2k-fast-q8` | 133 | `ollama/qwen25-coder-7b-fable5-2k-fast-q8` | 48.1% | 54.1% | 100.0% | 3 | 2 | 50.7 | V1 rerun reproduced the same pass rate 2 |
| `2026-07-09-19-54-59--full-ft-v2-qwen25-fable5-5k-fast-q8` | 133 | `ollama/qwen25-coder-7b-fable5-5k-fast-q8` | 40.6% | 48.1% | 100.0% | 13 | 0 | 44.9 | Larger 5k v2 model regressed on this benchmark |
| `2026-07-10-03-48-54--full-ft-v15-qwen25-fable5-3k-clean-q8` | 133 | `ollama/qwen25-coder-7b-fable5-3k-clean-q8` | 45.9% | 53.4% | 100.0% | 4 | 2 | 49.5 | Clean-filtered 3k v1.5 model beat the public base reference but did not beat v1 |
| `2026-07-11-07-05-14--full-ft-v3-nextcoder-python-fast-q8` | 133 | `ollama/qwen25-coder-7b-nextcoder-python-v3-fast-q8` | 45.9% | 53.4% | 100.0% | 4 | 2 | 51.7 | NextCoder Python edit-data v3 tied local base and v1.5, below v1 |
| `2026-07-12-00-06-41--full-ft-v4-mixed-nextcoder-fable-q8` | 133 | `ollama/qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 45.9% | 55.6% | 100.0% | 3 | 2 | 51.5 | Mixed 80% NextCoder fast + 20% Fable v1 fast; new best local score |
| `2026-07-12-02-36-56--full-ft-v4-rerun-mixed-nextcoder-fable-q8` | 133 | `ollama/qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 45.9% | 52.6% | 100.0% | 3 | 3 | 51.5 | V4 rerun regressed below local base/v1; first v4 run was not stable |
| `2026-07-12-17-54-08--full-ft-v4-rerun2-mixed-nextcoder-fable-q8` | 133 | `ollama/qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 45.9% | 52.6% | 100.0% | 5 | 2 | 52.1 | Second v4 rerun matched the first rerun aggregate; first v4 run looks like an outlier |
| `2026-07-21-01-40-17--full-ft-v5-verified-repair-q8` | 133 | `ollama/qwen25-coder-7b-v5-verified-repair-q8` | 44.4% | 51.1% | 100.0% | 3 | 2 | 91.0 | Base-model SFT on verified solutions/repairs caused capability forgetting |
| `2026-07-21-06-01-52--full-ft-v6-v1-repair-step16-q8` | 133 | `ollama/qwen25-coder-7b-v6-v1-repair-step16-q8` | 48.9% | 53.4% | 100.0% | 6 | 2 | 49.5 | Gentle v1 continuation on one-turn repair rows improved pass@1 but not pass@2 |
| `2026-07-21-08-27-39--full-ft-v7-v1-retry-q8` | 133 | `ollama/qwen25-coder-7b-v7-v1-retry-q8` | 48.1% | **56.4%** | 100.0% | 1 | 2 | 49.3 | Retry-shaped completion-only continuation; new best single local score |
| `2026-07-21-10-24-17--full-ft-v7-rerun-v1-retry-q8` | 133 | `ollama/qwen25-coder-7b-v7-v1-retry-q8` | 46.6% | 54.9% | 100.0% | 5 | 3 | 50.9 | Confirmation rerun; v7 two-run mean is 55.64% |

The fine-tuned model's full local old-harness score is `54.1%` pass rate after 2 tries, which is `+2.2` percentage points over the public `51.9%` reference row. Treat this as comparable but not identical: the run used the same old aider version, edit format, model API style, and 133-task corpus shape, but local Ollama/runtime details and the dirty benchmark worktree can still differ from the original public run.

The local base run scored `53.4%` pass rate after 2 tries, which is `+1.5` percentage points over the public `51.9%` reference row. This makes the local baseline stronger than the public row on this machine/run, and reduces v1's measured local lift to `+0.7` points over the local base (`54.1%` vs `53.4%`). The base model was also much slower in this run (`93.8` seconds/case) than the fine-tuned runs (`49.5` to `51.0` seconds/case), despite using the same harness settings.

The larger 5k v2 model scored `48.1%`, which is `-6.0` percentage points versus the 2k v1 model and `-3.8` percentage points versus the public base reference. More data did not improve this benchmark with the same filtering and hyperparameters; v1 remains the best benchmarked local model so far.

The v1 rerun reproduced `54.1%` pass rate 2 exactly, with `2` test timeouts instead of `3` and fewer user asks (`3` vs `11`). This makes the v1 score look stable rather than lucky, and makes the v2 regression more likely to be real.

The v1.5 clean-filtered run scored `53.4%` pass rate 2, which ties the local base run, beats the public `51.9%` base reference by `+1.5` points, and is `-0.7` points versus the v1 reproduced `54.1%`. The stricter data filter improved over v2 but did not replace v1 as the current best model.

The v3 NextCoder Python fast run also scored `53.4%` pass rate 2. It tied the local base and v1.5, beat the public `51.9%` reference by `+1.5` points, and remained `-0.7` points behind the reproduced v1 score. It was benchmarkable and well-formed, but the NextCoder-only edit-data training did not improve the old-aider score over the local base.

The first v4 mixed run scored `55.6%` pass rate 2, making it the best single local run so far. It was `+2.2` points over the local base and v3 (`53.4%`), `+1.5` points over the reproduced v1 score (`54.1%`), and `+3.7` points over the public old leaderboard reference (`51.9%`). Two fresh v4 reruns both scored `52.6%`, though, so the first v4 score is best treated as an outlier. V4 is not a reliable 4o-mini-beating model in its current form.

V5 introduced completion-only loss, token-safe rows, parent-level validation splitting, and execution-verified MBPP/QuixBugs/BigCodeBench examples. Training that corpus from the base model scored only `51.1%`: it gained four tasks over stable v1 but lost eight. V6 instead continued from the stable v1 adapter with 256 repair-only rows at `1e-6`; its halfway checkpoint scored `53.4%`.

V7 changed the continuation data to the actual retry shape: task and tests, a failed assistant whole-file response, test failure output, then the corrected whole-file completion. It used 128 verified repair rows, completion-only loss, `max_seq_length=1024`, and a `5e-7` learning rate for 16 steps. Its first full run scored `56.4%` (`75/133`), exceeding the previous `55.6%` record. A confirmation run scored `54.9%` (`73/133`), giving a two-run mean of `55.64%`. V7 is the new best single-run model and has a better two-run mean than v1, but still has material task-level variance.

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
| `fable5-5k-fast-lora` | `fable5_5k_fast` | 4900 | 100 | 2048 | 613 | 3478s resumed segment | 0.586 resumed avg | 1.237 | `models/qwen25-coder-7b-fable5-5k-fast-lora` |
| `fable5-3k-clean-lora` | `fable5_3k_clean` | 2940 | 60 | 2048 | 368 | TBD | n/a | 1.383 | `models/qwen25-coder-7b-fable5-3k-clean-lora` |
| `nextcoder-python-v3-fast-lora` | `nextcoder_python_v3_fast` | 1960 | 40 | 1024 | 245 | 3402s | 0.5065 | 0.4478 | `models/qwen25-coder-7b-nextcoder-python-v3-fast-lora` |
| `mixed-nextcoder-fable-v4-lora` | `mixed_nextcoder_fable_v4` | 2000 | 40 | 1024 | 250 | 3107s | 0.6584 | 0.7375 | `models/qwen25-coder-7b-mixed-nextcoder-fable-v4-lora` |
| `v5-verified-repair-lora` | `v5_verified_repair` | 909 | 43 | 1024 | 114 | 807s | 0.4984 | 0.6011 | `models/qwen25-coder-7b-v5-verified-repair-lora` |
| `v6-v1-repair-continuation-lora` | `v6_v1_repair_continuation` | 256 | 23 | 1024 | 32 | 257s | 0.4081 | 0.5480 | `models/qwen25-coder-7b-v6-v1-repair-continuation-lora` |
| `v7-v1-retry-continuation-lora` | `v7_v1_retry_continuation` | 128 | 23 | 1024 | 16 | 149s | 0.3505 | 0.4566 | `models/qwen25-coder-7b-v7-v1-retry-continuation-lora` |

v2 dataset search note: Hugging Face had newer Fable-related datasets after the first run, including `Glint-Research/Fable-5-traces`, `HelioAI/Fable-5-Distill-5500x`, and `WithinUsAI/fable_5_distillation_merged_cleaned_25k`. For this v2 run, `Nexlab/fable5-agentic-coding-sft` remained the practical source because it is MIT-licensed, coding-agent focused, and SFT-ready. Glint's rereleased traces are interesting but AGPL-licensed and raw trace-oriented; Helio/WithinUs samples inspected were generic reasoning/math rather than directly aligned coding-agent SFT data.

The v2 `fable5_5k_fast` split was prepared with:

```bash
python scripts/prepare_fable5_sft.py \
  --dataset Nexlab/fable5-agentic-coding-sft \
  --output-dir data/processed/fable5_5k_fast \
  --max-examples 5000 \
  --max-chars 12000
```

The v2 training run hit a command timeout after checkpoint `300`, then resumed successfully from `models/qwen25-coder-7b-fable5-5k-fast-lora/checkpoint-300` and completed all `613` steps. Final checkpoint eval loss was `1.237` at step `613`.

The v1.5 clean split was prepared with stricter code-edit filtering to reduce broad app/game/site generation examples and long tool traces:

```bash
python scripts/prepare_fable5_sft.py \
  --dataset Nexlab/fable5-agentic-coding-sft \
  --output-dir data/processed/fable5_3k_clean \
  --max-examples 3000 \
  --max-chars 10000 \
  --code-edit-clean \
  --max-tool-calls 8 \
  --max-assistant-chars 7000
```

The v1.5 training run completed all `368` steps from the base model with a lower learning rate (`1e-5`). Final checkpoint eval loss was `1.383` at step `368`, close to v1's final eval loss (`1.384`).

The v1.5 old-aider smoke and full benchmark completed successfully after export to `qwen25-coder-7b-fable5-3k-clean-q8`. Full benchmark result: `45.9%` pass rate 1, `53.4%` pass rate 2, `100.0%` well-formed, `4` user asks, `2` test timeouts, `49.5` seconds/case.

The v3 NextCoder Python fast split was prepared with:

```bash
python scripts/prepare_nextcoder_sft.py \
  --output-dir data/processed/nextcoder_python_v3_fast \
  --max-examples 2000 \
  --max-prompt-chars 4000 \
  --max-completion-chars 3500 \
  --min-completion-chars 120
```

The v3 fast training run completed all `245` steps with `max_seq_length=1024`, final train loss `0.5065`, and final eval loss `0.4478`. After Q8 GGUF export and Ollama import as `qwen25-coder-7b-nextcoder-python-v3-fast-q8`, direct smoke returned `ready` and old-aider one-case smoke passed `spiral-matrix` with `100.0%` well-formed output. The full benchmark resumed from an interrupted 28-case run and completed all `133` cases: `45.9%` pass rate 1, `53.4%` pass rate 2, `100.0%` well-formed, `4` user asks, `2` test timeouts, `51.7` seconds/case.

The v4 mixed split was prepared from existing processed data with:

```bash
python scripts/prepare_mixed_sft.py \
  --output-dir data/processed/mixed_nextcoder_fable_v4 \
  --total-train 2000 \
  --total-validation 40 \
  --fable-ratio 0.20
```

The resulting split contains `1600` NextCoder Python fast training rows and `400` Fable v1 fast training rows, with `32` NextCoder and `8` Fable validation rows. The v4 training run completed all `250` steps with `max_seq_length=1024`, final train loss `0.6584`, and final eval loss `0.7375`. The first Q8 export attempt failed because the C: drive had only `5.4G` free and the intermediate BF16 GGUF write was short. Removing redundant merged HF export directories recovered enough space, and the retry completed. After Ollama import as `qwen25-coder-7b-mixed-nextcoder-fable-v4-q8`, direct smoke returned `ready`, one-case old-aider smoke passed `matching-brackets`, and the first full 133-case benchmark scored `45.9%` pass rate 1, `55.6%` pass rate 2, `100.0%` well-formed, `3` user asks, `2` test timeouts, `51.5` seconds/case. Two fresh full reruns both scored `45.9%` pass rate 1 and `52.6%` pass rate 2, with `100.0%` well-formed output. The reruns had `3` and `5` user asks, `3` and `2` test timeouts, and `51.5` and `52.1` seconds/case respectively.
