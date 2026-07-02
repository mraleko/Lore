# Local Coding Model Fine-Tuning

This repo is for local QLoRA fine-tuning of `Qwen2.5-Coder-7B-Instruct` with Unsloth, using Fable 5-style coding-agent traces from Hugging Face and evaluating improvement with aider or coding benchmarks.

## Machine Profile

Detected environment:

- GPU: NVIDIA RTX 3060 Ti, 8 GB VRAM
- CPU: AMD Ryzen 5 5600X, 12 threads
- WSL memory: about 11 GB RAM, 4 GB swap
- Python: `python3 3.12.3`
- CUDA driver visible through `nvidia-smi`
- No local `nvcc` required for normal Unsloth/PyTorch wheel usage

This is enough for conservative 4-bit QLoRA on a 7B model. Start with sequence length `4096`, batch size `1`, and gradient accumulation.

## Repo Structure

```text
.
├── configs/
│   └── qwen25_coder_7b_lora.json
├── data/
│   ├── raw/          # ignored: optional downloaded/source data
│   └── processed/    # ignored: generated train/validation jsonl
├── docs/
│   └── benchmarks.md
├── models/           # ignored: adapters, merged models, GGUF exports
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_fable5_sft.py
│   └── train_unsloth_lora.py
└── requirements.txt
```

## Setup

Python 3.10 or 3.11 is preferred for Unsloth. The system currently has Python 3.12, which may work for some packages but is more likely to hit compatibility issues with training libraries.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For dataset inspection and conversion only:

```bash
pip install -r requirements-data.txt
```

For training, install PyTorch/Unsloth using the current Unsloth guidance for your CUDA driver, or install this repo's training dependencies:

```bash
pip install -r requirements.txt
```

If Unsloth publishes a newer install command for your CUDA/PyTorch combination, prefer the official command from `https://docs.unsloth.ai/`.

Check the environment:

```bash
python scripts/check_env.py
```

The full training install is large because it includes PyTorch, CUDA runtime wheels, Triton, xFormers, bitsandbytes, and Unsloth.

## Dataset Discovery

Recommended starting dataset:

- `Nexlab/fable5-agentic-coding-sft`: public, MIT-licensed, about 160k curated SFT examples, OpenAI-style `messages`, includes coding-agent tool-use loops.

Secondary dataset:

- `Glint-Research/Fable-5-traces`: public, AGPL-3.0, smaller/specialized trace corpus. Useful for experiments, but less straightforward as a first SFT source.

Other likely Fable 5 datasets on Hugging Face:

- `Glint-Research/Fable-5-traces`
- `kelexine/fable-5-sft-traces`
- `Swarm-AI-Research/fable5-traces-sft`
- `AlinCiocan/fable-5-claude-code-traces`

Start by inspecting schema:

```bash
python scripts/inspect_dataset.py Nexlab/fable5-agentic-coding-sft
```

If that dataset is unavailable or too narrow, inspect the primary traces dataset:

```bash
python scripts/inspect_dataset.py Glint-Research/Fable-5-traces
```

## Prepare SFT Data

Convert a dataset split into Qwen-style chat JSONL:

```bash
python scripts/prepare_fable5_sft.py \
  --dataset Nexlab/fable5-agentic-coding-sft \
  --output-dir data/processed/fable5 \
  --max-examples 20000
```

The output format is one JSON object per line:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

For Fable 5 tool-use examples, `tool_calls`, `tool_call_id`, and `name` fields are preserved when present.

## Train

Run a small first training job:

```bash
python scripts/train_unsloth_lora.py --config configs/qwen25_coder_7b_lora.json
```

The default config is intentionally conservative for 8 GB VRAM.

## Benchmark

See `docs/benchmarks.md` for the baseline/fine-tuned evaluation flow.
