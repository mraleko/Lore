#!/usr/bin/env python3
"""Export an Unsloth/PEFT adapter to GGUF for llama.cpp/Ollama use."""

from __future__ import annotations

import argparse

from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/qwen25-coder-7b-fable5-2k-fast-lora", help="Adapter or model directory")
    parser.add_argument("--output-dir", default="models/qwen25-coder-7b-fable5-2k-fast-gguf")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--quantization", default="q8_0", help="Unsloth GGUF quantization method")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    if not hasattr(model, "save_pretrained_gguf"):
        raise SystemExit("This Unsloth model instance does not expose save_pretrained_gguf().")

    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method=args.quantization)
    print(f"Exported GGUF to {args.output_dir}")


if __name__ == "__main__":
    main()
