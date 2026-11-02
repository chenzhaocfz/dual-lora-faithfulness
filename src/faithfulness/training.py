"""QLoRA fine-tuning for every adapter role (Unsloth + Hugging Face Trainer).

Loss is applied to completion tokens only: a Solver is trained to produce A, not to
reproduce the reasoning trace it was given. Prompt and completion token counts are
recorded because the roles have very different input/output lengths.
"""
# Unsloth must be imported before transformers / peft
from unsloth import FastLanguageModel, is_bfloat16_supported

import gc
import inspect
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from . import config
from .data import load_json, save_json


def load_base_for_training(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_name),
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **config.LORA_CONFIG)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    return model, tokenizer


def tokenize_examples(examples, tokenizer):
    """[{prompt, completion}] -> Dataset(input_ids, attention_mask, labels) with prompt tokens masked."""
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    prompt_tokens = completion_tokens = 0
    for ex in examples:
        p = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
        c = tokenizer(ex["completion"], add_special_tokens=False).input_ids
        rows["input_ids"].append(p + c)
        rows["attention_mask"].append([1] * (len(p) + len(c)))
        rows["labels"].append([-100] * len(p) + c)
        prompt_tokens += len(p)
        completion_tokens += len(c)
    token_stats = {
        "examples": len(examples),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,  # tokens that receive loss
        "total_tokens": prompt_tokens + completion_tokens,
        "max_example_tokens": max(len(ids) for ids in rows["input_ids"]),
    }
    return Dataset.from_dict(rows), token_stats


def train_adapter(role, dataset_path, output_dir, base_model=config.BASE_MODEL, max_steps=-1, resume=True,
                  **train_overrides):
    """Train one LoRA adapter on the 4-bit base model and write training_summary.json next to it.

    Adapter-only checkpoints are written to output_dir/checkpoints every save_steps, so the Trap Test
    can be run at points during training as well as on the final adapter.

    max_steps > 0 gives a short dry run (useful for checking memory on the 4070).
    resume=True continues from the latest checkpoint in output_dir/checkpoints if a run was
    interrupted (optimizer, LR schedule and data order are restored). Delete that folder,
    or pass resume=False, to start over after changing data or settings.
    """
    output_dir = Path(output_dir)
    print("=" * 70)
    print(f"TRAINING {role.upper()} ADAPTER")
    print(f"Base:    {base_model}")
    print(f"Data:    {dataset_path}")
    print(f"Output:  {output_dir}")
    print("=" * 70)

    model, tokenizer = load_base_for_training(base_model)
    examples = load_json(dataset_path)
    dataset, token_stats = tokenize_examples(examples, tokenizer)
    if token_stats["max_example_tokens"] > config.MAX_SEQ_LENGTH:
        raise ValueError(f"An example has {token_stats['max_example_tokens']} tokens "
                         f"(> MAX_SEQ_LENGTH={config.MAX_SEQ_LENGTH}); re-run the dataset notebook filters.")
    print(json.dumps(token_stats, indent=2))

    train_config = {**config.TRAIN_CONFIG, **train_overrides}
    if "warmup_ratio" not in inspect.signature(TrainingArguments.__init__).parameters:
        # transformers 5 removed warmup_ratio; warmup_steps < 1 is read as a ratio instead
        train_config["warmup_steps"] = train_config.pop("warmup_ratio")
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=max_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
        remove_unused_columns=False,
        **train_config,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    checkpoint = get_last_checkpoint(args.output_dir) if resume and Path(args.output_dir).is_dir() else None
    if checkpoint:
        print(f"Resuming from {checkpoint}")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    runtime = time.time() - start

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    summary = {
        "role": role,
        "base_model": str(base_model),
        "dataset": str(dataset_path),
        "lora": config.LORA_CONFIG,
        "training": {**train_config, "max_steps": max_steps},
        "effective_batch_size": train_config["per_device_train_batch_size"] * train_config["gradient_accumulation_steps"],
        "tokens": token_stats,
        "global_steps": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "final_logged_loss": losses[-1] if losses else None,
        # Mean of the logged losses over the whole run. Trainer's own training_loss is wrong after a
        # resume: it sums only the resumed steps but divides by all of them.
        "mean_train_loss": sum(losses) / len(losses) if losses else result.training_loss,
        # When resumed, runtime covers only the steps run after the checkpoint
        "resumed_from_checkpoint": checkpoint,
        "runtime_seconds": round(runtime, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
        "log_history": trainer.state.log_history,
    }
    save_json(summary, output_dir / "training_summary.json")
    print(f"Saved adapter and training_summary.json to {output_dir}")
    print(f"Steps: {summary['global_steps']} | mean loss: {summary['mean_train_loss']:.4f} | "
          f"peak VRAM: {summary['peak_vram_gb']} GB | {runtime / 60:.1f} min")

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return summary
