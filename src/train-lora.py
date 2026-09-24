# ==============================================================================
# 1. Unsloth MUST be imported first before trl, transformers, peft, or torch!
# This allows Unsloth to patch memory kernels and prevent the <EOS_TOKEN> bug.
# ==============================================================================
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import json
import argparse
import time
import inspect
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Map adapters to their respective datasets and target directories (Page 11 & 14)
ADAPTER_CONFIGS = {
    "monolithic": {
        "dataset_path": "data/processed/baseline_monolithic.json",
        "output_dir": "./finetuned_adapters/adapter_monolithic",
        "description": "Monolithic Control Model (CoT + Final Answer)"
    },
    "thinker": {
        "dataset_path": "data/processed/thinker_only.json",
        "output_dir": "./finetuned_adapters/adapter_thinker",
        "description": "Dual-LoRA Thinker Adapter (CoT Reasoning Trace only)"
    },
    "solver": {
        "dataset_path": "data/processed/solver_only.json",
        "output_dir": "./finetuned_adapters/adapter_solver",
        "description": "Dual-LoRA Solver Adapter (Generates final answer from CoT)"
    }
}

def format_dataset_to_chatml(json_path: str, tokenizer):
    """Loads processed JSON and applies Qwen ChatML conversational structure."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing dataset: {json_path}! Run generate_synthetic_data.py first.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_conversations = []
    for item in data:
        conversation = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]
        
        # Apply standard ChatML template (<|im_start|>...<|im_end|>)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(conversation, tokenize=False)
        else:
            text = (
                f"<|im_start|>system\n{item['instruction']}<|im_end|>\n"
                f"<|im_start|>user\n{item['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['output']}<|im_end|>"
            )
        formatted_conversations.append(text)

    return Dataset.from_dict({"text": formatted_conversations})

def train_single_adapter(adapter_type: str, epochs: int, lr: float, max_steps: int = -1):
    cfg = ADAPTER_CONFIGS[adapter_type]
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING: {adapter_type.upper()} ADAPTER")
    print(f"Description: {cfg['description']}")
    print(f"Dataset:     {cfg['dataset_path']}")
    print(f"Output:      {cfg['output_dir']}")
    print("=" * 70)

    # 1. Load Base Model in 4-bit (Unsloth QLoRA)
    MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    MAX_LENGTH = 2048

    print(f"\n[1/4] Loading {MODEL_NAME} (4-bit QLoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Ensure valid padding and EOS tokens for Qwen
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Inject LoRA Adapters (Page 12 Thesis Specifications)
    print("\n[2/4] Applying LoRA PEFT target modules (r=16, alpha=16)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load & Format Dataset
    print(f"\n[3/4] Preparing ChatML training data...")
    dataset = format_dataset_to_chatml(cfg["dataset_path"], tokenizer)
    print(f"Total training samples: {len(dataset)}")

    # 4. Configure SFTConfig dynamically
    print("\n[4/4] Initializing SFT Trainer...")
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    length_arg_name = "max_length" if "max_length" in sft_config_params else "max_seq_length"

    config_kwargs = {
        "output_dir": cfg["output_dir"],
        "dataset_text_field": "text",
        length_arg_name: MAX_LENGTH,
        "packing": False,
        "dataset_num_proc": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,  # Effective batch size = 8
        "warmup_ratio": 0.05,
        "num_train_epochs": epochs if max_steps == -1 else 1,
        "max_steps": max_steps,
        "learning_rate": lr,
        "logging_steps": 5,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 3407,
        "report_to": "none",
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
    }

    # Pass actual EOS token if supported by SFTConfig
    if "eos_token" in sft_config_params:
        config_kwargs["eos_token"] = tokenizer.eos_token

    training_args = SFTConfig(**config_kwargs)

    # Bug #2797 Safeguard: Override any placeholder injected by patching
    if getattr(training_args, "eos_token", None) == "<EOS_TOKEN>":
        training_args.eos_token = tokenizer.eos_token

    # Configure SFTTrainer args
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    # 5. Train
    print("\nStarting training loop...")
    start_train_time = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - start_train_time

    # 6. Save Adapter Locally
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE IN {elapsed:.2f}s! SAVING ADAPTER...")
    print("=" * 70)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"✓ Adapter weights & tokenizer saved locally to: {cfg['output_dir']}")

    # Cleanup memory
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 adapters locally using Unsloth QLoRA.")
    parser.add_argument(
        "--adapter",
        type=str,
        default="monolithic",
        choices=["monolithic", "thinker", "solver", "all"],
        help="Which adapter to train: 'monolithic', 'thinker', 'solver', or 'all'."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4).")
    parser.add_argument("--steps", type=int, default=-1, help="Force max steps for debugging (default: -1 for full dataset).")
    args = parser.parse_args()

    targets = ["monolithic", "thinker", "solver"] if args.adapter == "all" else [args.adapter]

    for target in targets:
        train_single_adapter(
            adapter_type=target,
            epochs=args.epochs,
            lr=args.lr,
            max_steps=args.steps
        )

if __name__ == "__main__":
    main()