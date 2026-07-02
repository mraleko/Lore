#!/usr/bin/env python3
"""Train a Qwen2.5-Coder LoRA adapter with Unsloth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen25_coder_7b_lora.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 3407),
    )

    data_files = {"train": cfg["train_file"]}
    validation_file = cfg.get("validation_file")
    if validation_file and Path(validation_file).exists():
        data_files["validation"] = validation_file
    dataset = load_dataset("json", data_files=data_files)

    def formatting_prompts_func(batch: dict[str, Any]) -> list[str]:
        texts = []
        for messages in batch["messages"]:
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return texts

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        report_to=cfg.get("report_to", "none"),
        fp16=False,
        bf16=False,
        eval_strategy="steps" if "validation" in dataset else "no",
        save_strategy="steps",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        formatting_func=formatting_prompts_func,
        max_seq_length=cfg["max_seq_length"],
        packing=False,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
