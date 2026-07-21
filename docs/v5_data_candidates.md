# V5 Data Candidates

Goal: improve stable old-aider Exercism Python performance by training exact small-task correctness and second-attempt repair behavior, without training on the actual aider benchmark tasks.

## Best Candidates

### MBPP

Dataset: `google-research-datasets/mbpp`

License: `cc-by-4.0`

Why it fits:

- Small Python programming tasks.
- Includes reference code and assert-style tests.
- Similar size/shape to Exercism without being the same corpus.
- Easy to convert into both full-file solution examples and repair examples.

Recommended use:

- Use `full/train` and `full/validation`, not `test`, to avoid contaminating common secondary evals.
- Generate prompts that include task text and tests.
- Completion should be only the Python code.
- For repair training, mutate reference solutions with controlled bugs, run tests to confirm failure, then train from bad code plus failing tests to the original passing solution.

Risks:

- Public benchmark dataset. Avoid using it if MBPP remains a target eval.
- Some examples are lower quality or underspecified; verify generated files pass included tests before training.

### QuixBugs Python

Dataset: `Muennighoff/quixbugs`

License: dataset card does not expose a clear license in the API response; verify before redistribution.

Why it fits:

- Has `buggy_program`, `solution`, docstring, and tests.
- Directly aligned with second-attempt repair.
- Compact algorithmic bugs: good fit for 1024-token training.

Recommended use:

- Use as high-value repair examples, not as the bulk of training because it is small.
- Format as: user sees buggy code, tests, failing intent; assistant returns corrected full file only.

Risks:

- Very small, so it can overfit style if overweighted.
- Need license verification.

### BigCodeBench

Dataset: `bigcode/bigcodebench`

License: `apache-2.0`

Why it fits:

- Includes instruction prompts, canonical solutions, and unittest code.
- Strong test-backed signal.
- Useful for exact API/spec compliance.

Recommended use:

- Use lightly and filter hard:
  - Python only.
  - Short prompts and short solutions.
  - Standard-library-only tasks where possible.
  - Exclude examples requiring heavy external libraries or randomness.
- Prefer solution examples over repair examples unless we generate controlled mutations.

Risks:

- More library/API-heavy than Exercism.
- Can push model toward function-completion style rather than repository editing if overused.

### APPS

Dataset: `codeparrot/apps`

License: `mit`

Why it fits:

- Large Python problem/test corpus.
- Useful source of algorithmic correctness examples.

Recommended use:

- Only use the easy/introductory subset if we load it.
- Filter aggressively for short statements, short solutions, and simple stdin/stdout-free functions if possible.

Risks:

- Loader requires special handling because the dataset script is deprecated in current `datasets`.
- Many examples are competitive-programming style, less like Exercism full-file edits.

## Lower-Priority Candidates

### CodeFeedback

Datasets:

- `m-a-p/CodeFeedback-Filtered-Instruction`
- `Leon-Leee/Code-Feedback-decontamination`

License: `apache-2.0`

Why it may help:

- Execution/refinement oriented.
- Large enough to filter for Python.

Why not primary:

- Broad instruction data, not consistently test-backed.
- Similar risk to Fable/NextCoder bulk data: plausible code style without exact benchmark improvement.

Recommended use:

- At most a small filtered supplement.
- Keep only rows with Python, explicit tests/errors, and concise corrected code.

### Python Debugging Synthetic

Dataset: `creeperdatasets/python_debugging`

License: `mit`

Why it may help:

- Simple bug-fix input/output pairs.

Why not primary:

- Only 75 rows.
- Outputs include explanation, not just corrected code.
- Synthetic and sometimes underspecified.

Recommended use:

- Use as format inspiration or tiny supplement after rewriting completions to code-only.

### SWE-smith / SWE-bench Trajectories

Dataset: `SWE-bench/SWE-smith-trajectories`

License: `mit`

Why it may help:

- Real agentic repair trajectories.

Why not primary for v5:

- Repo-scale, long-context, tool-heavy traces.
- Not aligned with 1024-token small-task old-aider benchmark.
- Local sample loading was unstable enough to avoid making this a dependency for the next run.

Recommended use:

- Skip for v5 or use only a tiny filtered patch-only supplement later.

## Recommended V5 Recipe

Do not train on actual aider Exercism benchmark tasks.

Target size: `2500-3000` examples.

Recommended mix:

| Source | Share | Purpose |
| --- | ---: | --- |
| MBPP verified solution examples | 35% | small task exactness |
| MBPP synthetic repair examples | 30% | failing-code to passing-code repair |
| QuixBugs repair examples | 10% | real buggy-code repair patterns |
| BigCodeBench filtered short examples | 15% | richer tests/spec compliance |
| Fable v1 fast traces | 10% | preserve aider/agentic editing behavior |

Training shape:

- `max_seq_length=1024`
- `learning_rate=5e-6` or lower
- one epoch
- QLoRA rank 16, same as v4
- completion should be code-only whenever possible

Expected improvement target:

- Raise stable second-attempt repair, not pass@1 alone.
- Watch tasks like `change`, `word-count`, `pascals-triangle`, `triangle`, `saddle-points`, `house`, `ocr-numbers`, and `high-scores` because these explain much of v4's variance.
