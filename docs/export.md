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
