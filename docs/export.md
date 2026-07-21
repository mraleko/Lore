# Export And Serve

The first trained adapter is saved at:

```text
models/qwen25-coder-7b-fable5-2k-fast-lora
```

Export to GGUF for Ollama/llama.cpp:

```bash
python scripts/export_unsloth_gguf.py \
  --model models/qwen25-coder-7b-fable5-2k-fast-lora \
  --output-dir models/qwen25-coder-7b-fable5-2k-fast-gguf \
  --quantization q8_0
```

After export, create an Ollama model from the GGUF with a `Modelfile`, then benchmark it with the same aider command pattern used for the local baseline.

Created local Ollama model:

```bash
ollama create qwen25-coder-7b-fable5-2k-fast-q8 \
  -f models/qwen25-coder-7b-fable5-2k-fast-gguf_gguf/Modelfile
```

Smoke test:

```bash
ollama run qwen25-coder-7b-fable5-2k-fast-q8 "Reply with exactly: ready"
```

The model responded with `ready`.

## V7 V1 Retry Continuation

The best single-run adapter is saved at:

```text
models/qwen25-coder-7b-v7-v1-retry-continuation-lora
```

The final Q8 GGUF is:

```text
models/qwen25-coder-7b-v7-v1-retry-gguf_gguf/qwen2.5-coder-7b-instruct.Q8_0.gguf
```

Created Ollama model:

```text
qwen25-coder-7b-v7-v1-retry-q8
```

Its full old-aider runs scored `56.4%` and `54.9%` pass@2. The first is the highest local score so far; the two-run mean is `55.64%`.

## V4 Mixed NextCoder/Fable Adapter

The mixed v4 adapter is saved at:

```text
models/qwen25-coder-7b-mixed-nextcoder-fable-v4-lora
```

Export command:

```bash
python scripts/export_unsloth_gguf.py \
  --model models/qwen25-coder-7b-mixed-nextcoder-fable-v4-lora \
  --output-dir models/qwen25-coder-7b-mixed-nextcoder-fable-v4-gguf \
  --quantization q8_0
```

Exported Q8 GGUF:

```text
models/qwen25-coder-7b-mixed-nextcoder-fable-v4-gguf_gguf/qwen2.5-coder-7b-instruct.Q8_0.gguf
```

Created local Ollama model:

```bash
ollama create qwen25-coder-7b-mixed-nextcoder-fable-v4-q8 \
  -f models/qwen25-coder-7b-mixed-nextcoder-fable-v4-gguf_gguf/Modelfile
```

Smoke test:

```bash
ollama run qwen25-coder-7b-mixed-nextcoder-fable-v4-q8 "Reply with exactly: ready"
```

The model responded with `ready`.

The first v4 export attempt failed because the C: drive had only `5.4G` free and the intermediate BF16 GGUF write was short. Removing redundant merged HF export directories recovered enough room for the retry. Keep LoRA directories and final `_gguf` directories; the non-`_gguf` merged HF export directories can be removed after successful Ollama import if disk space is tight.

## V3 NextCoder Python Fast Adapter

The NextCoder Python edit-data v3 fast adapter is saved at:

```text
models/qwen25-coder-7b-nextcoder-python-v3-fast-lora
```

Export command:

```bash
python scripts/export_unsloth_gguf.py \
  --model models/qwen25-coder-7b-nextcoder-python-v3-fast-lora \
  --output-dir models/qwen25-coder-7b-nextcoder-python-v3-fast-gguf \
  --quantization q8_0
```

Exported Q8 GGUF:

```text
models/qwen25-coder-7b-nextcoder-python-v3-fast-gguf_gguf/qwen2.5-coder-7b-instruct.Q8_0.gguf
```

Created local Ollama model:

```bash
ollama create qwen25-coder-7b-nextcoder-python-v3-fast-q8 \
  -f models/qwen25-coder-7b-nextcoder-python-v3-fast-gguf_gguf/Modelfile
```

Smoke test:

```bash
ollama run qwen25-coder-7b-nextcoder-python-v3-fast-q8 "Reply with exactly: ready"
```

The model responded with `ready`.

## V1.5 Clean Adapter

The clean-filtered v1.5 adapter is saved at:

```text
models/qwen25-coder-7b-fable5-3k-clean-lora
```

Export command:

```bash
python scripts/export_unsloth_gguf.py \
  --model models/qwen25-coder-7b-fable5-3k-clean-lora \
  --output-dir models/qwen25-coder-7b-fable5-3k-clean-gguf \
  --quantization q8_0
```

Exported Q8 GGUF:

```text
models/qwen25-coder-7b-fable5-3k-clean-gguf_gguf/qwen2.5-coder-7b-instruct.Q8_0.gguf
```

Created local Ollama model:

```bash
ollama create qwen25-coder-7b-fable5-3k-clean-q8 \
  -f models/qwen25-coder-7b-fable5-3k-clean-gguf_gguf/Modelfile
```

Smoke test:

```bash
ollama run qwen25-coder-7b-fable5-3k-clean-q8 "Reply with exactly: ready"
```

The model responded with `ready`.

The intermediate merged HF export directory `models/qwen25-coder-7b-fable5-2k-fast-gguf` was removed after Ollama import to recover disk space. The final GGUF directory and LoRA adapter were kept.

## V2 Adapter

The larger v2 adapter is saved at:

```text
models/qwen25-coder-7b-fable5-5k-fast-lora
```

Export command:

```bash
python scripts/export_unsloth_gguf.py \
  --model models/qwen25-coder-7b-fable5-5k-fast-lora \
  --output-dir models/qwen25-coder-7b-fable5-5k-fast-gguf \
  --quantization q8_0
```

Suggested Ollama model name after export:

```text
qwen25-coder-7b-fable5-5k-fast-q8
```

Created local Ollama model:

```bash
ollama create qwen25-coder-7b-fable5-5k-fast-q8 \
  -f models/qwen25-coder-7b-fable5-5k-fast-gguf_gguf/Modelfile
```

Smoke test:

```bash
ollama run qwen25-coder-7b-fable5-5k-fast-q8 "Reply with exactly: ready"
```

The model responded with `ready`.
