# Benchmark Analysis

This compares full old-aider `v0.56.0` runs on the same 133-task Exercism Python corpus with `whole` edit format and `ollama/...` model access.

## Runs Compared

| Label | Run | Model | Pass Rate 1 | Pass Rate 2 | Seconds/Case |
| --- | --- | --- | ---: | ---: | ---: |
| base | `2026-07-10-21-09-35--full-base-qwen25-coder-7b-instruct-q8` | `qwen2.5-coder:7b-instruct-q8_0` | 46.6% | 53.4% | 93.8 |
| v1 | `2026-07-08-23-43-49--full-ft-qwen25-fable5-2k-fast-q8` | `qwen25-coder-7b-fable5-2k-fast-q8` | 49.6% | 54.1% | 51.0 |
| v1 rerun | `2026-07-09-21-44-21--full-ft-v1-rerun-qwen25-fable5-2k-fast-q8` | `qwen25-coder-7b-fable5-2k-fast-q8` | 48.1% | 54.1% | 50.7 |
| v1.5 | `2026-07-10-03-48-54--full-ft-v15-qwen25-fable5-3k-clean-q8` | `qwen25-coder-7b-fable5-3k-clean-q8` | 45.9% | 53.4% | 49.5 |
| v2 | `2026-07-09-19-54-59--full-ft-v2-qwen25-fable5-5k-fast-q8` | `qwen25-coder-7b-fable5-5k-fast-q8` | 40.6% | 48.1% | 44.9 |
| v4 | `2026-07-12-00-06-41--full-ft-v4-mixed-nextcoder-fable-q8` | `qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 45.9% | 55.6% | 51.5 |
| v4 rerun | `2026-07-12-02-36-56--full-ft-v4-rerun-mixed-nextcoder-fable-q8` | `qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 45.9% | 52.6% | 51.5 |
| v4 rerun 2 | `2026-07-12-17-54-08--full-ft-v4-rerun2-mixed-nextcoder-fable-q8` | `qwen25-coder-7b-mixed-nextcoder-fable-v4-q8` | 45.9% | 52.6% | 52.1 |
| v5 | `2026-07-21-01-40-17--full-ft-v5-verified-repair-q8` | `qwen25-coder-7b-v5-verified-repair-q8` | 44.4% | 51.1% | 91.0 |
| v6 step 16 | `2026-07-21-06-01-52--full-ft-v6-v1-repair-step16-q8` | `qwen25-coder-7b-v6-v1-repair-step16-q8` | 48.9% | 53.4% | 49.5 |
| v7 | `2026-07-21-08-27-39--full-ft-v7-v1-retry-q8` | `qwen25-coder-7b-v7-v1-retry-q8` | 48.1% | **56.4%** | 49.3 |
| v7 rerun | `2026-07-21-10-24-17--full-ft-v7-rerun-v1-retry-q8` | `qwen25-coder-7b-v7-v1-retry-q8` | 46.6% | 54.9% | 50.9 |

## Main Takeaway

The v1 improvement over local base is real but small in this sample: `54.1%` versus `53.4%`, a net gain of one task out of 133. The strongest non-score signal is speed: v1 and v1.5 finish at about `50s/case`, while the local base run took `93.8s/case`.

v1 is stable across two runs at the aggregate level. Both v1 runs scored `54.1%` pass rate 2, and they disagreed on only two tasks: `circular-buffer` and `dnd-character`.

v4 is not stable at the first-run high score. Its first run scored `55.6%` pass rate 2, but two reruns both scored `52.6%`. All three v4 runs had the same pass rate 1 (`45.9%`), so the difference came from second-try recovery variance rather than worse first edits overall.

V7 produced the new highest score, `56.4%`, by solving 64 tasks on the first try and recovering 11 more on retry. Its confirmation run solved 62 on the first try and 11 on retry, scoring `54.9%`. The two-run mean is `55.64%`, but the 2-task aggregate spread means the improvement is promising rather than fully stable.

## V5-V7 Iteration

V5 trained 909 execution-verified MBPP, QuixBugs, and filtered BigCodeBench solution/repair rows from the base model. It used parent-disjoint validation, explicit token limits, and completion-only loss, fixing the truncation and full-sequence-loss problems in older runs. It nevertheless scored `51.1%`, gaining only `acronym`, `circular-buffer`, `resistor-color-trio`, and `satellite` over stable v1 while losing eight tasks. The focused corpus replaced too much of v1's useful behavior when trained from base.

V6 continued stable v1 on 256 repair-only rows at `1e-6`. The halfway checkpoint scored `53.4%`: versus stable v1 it gained `circular-buffer` and `satellite`, but lost `dnd-character`, `house`, and `yacht`. This reduced forgetting but did not improve second-attempt recovery.

V7 continued stable v1 for only 16 steps at `5e-7` on 128 rows formatted like aider's retry conversation. Versus stable v1, its record run gained four pass@2 tasks (`robot-simulator`, `satellite`, `word-count`, and `zebra-puzzle`) and lost one (`hexadecimal`), a net gain of three tasks. The confirmation rerun gained `circular-buffer` and `ocr-numbers` versus the first v7 run but lost `robot-simulator`, `word-count`, `yacht`, and `zebra-puzzle`, explaining the two-task score drop.

The strongest causal signal is retry formatting. One-turn verified repair SFT did not improve pass@2, while the smaller multi-turn retry continuation raised the record from 74 to 75 solved tasks and the two-run mean from v1's 72 to v7's 74.

## v4 Rerun Variance

The first v4 rerun lost six pass@2 tasks and gained two, for a net loss of four tasks (`74` solved down to `70` solved). None of the six lost tasks had test timeouts, malformed responses, syntax errors, indentation errors, or user asks. The second v4 rerun also solved `70` tasks. It swapped four tasks versus the first rerun (`house` and `ocr-numbers` regressed, while `pascals-triangle` and `saddle-points` recovered), but the aggregate score stayed identical.

| Task | Base | v1 | v1 Rerun | v1.5 | v3 | v4 | v4 Rerun | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `change` | FF | FF | FF | FF | FF | FP | FF | v4 first-run second-try win did not reproduce |
| `high-scores` | FP | FF | FF | FF | FP | FP | FF | unstable; base/v3 solve on second try |
| `pascals-triangle` | FF | P | P | FF | FF | P | FF | v1 stable, v4 unstable |
| `saddle-points` | FP | FP | FP | FP | FP | FP | FF | broadly second-try-sensitive |
| `triangle` | P | FP | FP | FP | FF | P | FF | usually solvable, but v4 rerun failed both tries |
| `word-count` | FF | FF | FF | FF | FP | FP | FF | v3/v4 first solve on second try, v4 rerun fails |
| `hexadecimal` | P | P | P | P | FP | FF | FP | rerun gained this task |
| `yacht` | P | P | P | P | P | FF | P | rerun gained this task |
| `palindrome-products` | FP | P | P | FP | P | FP | P | pass@1 improved, pass@2 unchanged |

Legend: `P` means passed on first try, `FP` means failed first try and passed second try, `FF` means failed both tries.

The biggest signal is that v4's pass@1 count was identical across all three runs (`61` tasks). The first run reached `74` pass@2 solves because it recovered `13` tasks on the second try; both reruns recovered only `9`. The 4-task score drop is exactly the lost second-try recovery margin. With two reruns landing at `52.6%`, the first `55.6%` run should be treated as an outlier rather than v4's expected performance.

## Pairwise Changes Versus Base

| Model | Wins Over Base | Losses Versus Base | Net |
| --- | ---: | ---: | ---: |
| v1 | 5 | 4 | +1 |
| v1 rerun | 5 | 4 | +1 |
| v1.5 | 3 | 3 | 0 |
| v2 | 4 | 11 | -7 |

Stable v1 wins over base in both v1 runs:

| Task | Base | v1 | v1 Rerun | Notes |
| --- | --- | --- | --- | --- |
| `affine-cipher` | FF | P | FP | v1/v1 rerun solve; base fails both tries |
| `anagram` | FF | P | P | stable first-try win |
| `pascals-triangle` | FF | P | P | stable first-try win |
| `say` | FF | FP | FP | stable second-try win |
| `twelve-days` | FF | P | P | stable first-try win; all fine-tunes solve |

Stable v1 losses versus base in both v1 runs:

| Task | Base | v1 | v1 Rerun | Notes |
| --- | --- | --- | --- | --- |
| `high-scores` | FP | FF | FF | base solves on second try; v1 fails both |
| `markdown` | P | FF | FF | base solves first try; v1 regresses |
| `satellite` | FP | FF | FF | base solves on second try; v1 fails both |

Flaky v1 tasks:

| Task | Base | v1 | v1 Rerun | Notes |
| --- | --- | --- | --- | --- |
| `circular-buffer` | P | P | FF | v1 rerun regressed only |
| `dnd-character` | FP | FF | FP | v1 first run regressed only |

Legend: `P` means passed on first try, `FP` means failed first try and passed second try, `FF` means failed both tries.

## v1.5 And v2 Signals

v1.5 ties local base at `53.4%`, but it solves different tasks. Its wins over base are `error-handling`, `run-length-encoding`, and `twelve-days`. Its losses are `high-scores`, `house`, and `rational-numbers`.

v2 is clearly worse on this benchmark. It wins `affine-cipher`, `killer-sudoku-helper`, `tournament`, and `twelve-days`, but loses 11 tasks versus base: `binary-search`, `house`, `isbn-verifier`, `largest-series-product`, `markdown`, `palindrome-products`, `rational-numbers`, `robot-name`, `satellite`, `secret-handshake`, and `square-root`.

## Difficulty Buckets

Solved by all five full runs: 57 tasks.

Never solved by any of the five full runs: 53 tasks.

Exactly one model solved these tasks:

| Task | Only Solver |
| --- | --- |
| `error-handling` | v1.5 |
| `killer-sudoku-helper` | v2 |
| `run-length-encoding` | v1.5 |
| `tournament` | v2 |

Base fails but at least one fine-tune solves:

| Task | Fine-Tune Solvers |
| --- | --- |
| `affine-cipher` | v1, v1 rerun, v2 |
| `anagram` | v1, v1 rerun |
| `error-handling` | v1.5 |
| `killer-sudoku-helper` | v2 |
| `pascals-triangle` | v1, v1 rerun |
| `run-length-encoding` | v1.5 |
| `say` | v1, v1 rerun |
| `tournament` | v2 |
| `twelve-days` | v1, v1 rerun, v1.5, v2 |

## Interpretation

The score gap is too small to claim a strong capability improvement. v1 is the best current model, but its local gain over base is one net task. The useful evidence is that v1 consistently shifts the same five tasks from fail to pass across two runs while consistently regressing three tasks.

The data-scaling experiment failed: v2 adds more data and drops to `48.1%`. The clean-filtering experiment recovered most of the loss but only ties base. More broad SFT data is unlikely to be the next best move.

## Recommended Next Step

Keep v7 as the current best model. If improving stability further, expand the parent-disjoint retry corpus rather than increasing learning rate or adding generic solution rows. Evaluate candidate checkpoints on a separate held-out repair set before another old-aider run, and use at least two full benchmark runs because second-attempt task flips remain substantial.
