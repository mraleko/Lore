# Benchmark Plan

The goal is to quantify whether Fable 5 agentic-coding fine-tuning improves a local Qwen2.5-Coder 7B model.

## Primary Benchmark: aider

Use aider as the main benchmark because it measures repository-editing behavior rather than isolated coding snippets.

Recommended flow:

1. Run the benchmark with the base model.
2. Train the LoRA adapter.
3. Run the same benchmark with the fine-tuned model.
4. Compare solve rate, edit validity, test pass rate, and regression patterns.

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
| baseline | Qwen2.5-Coder-7B-Instruct | none | 0 | n/a | n/a | aider | TBD | base score |
| fable5-sft-v1 | Qwen2.5-Coder-7B + LoRA | Fable 5 | TBD | 4096 | 16 | aider | TBD | first pass |
