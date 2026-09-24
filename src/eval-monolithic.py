import os
import json
import re
import argparse
import time
import torch
from unsloth import FastLanguageModel

def extract_numeric_answer(text: str) -> str:
    """Extracts numeric answer inside \\boxed{} or falls back to last number."""
    if not text:
        return ""
    # 1. Primary: Look for \boxed{...}
    boxed = re.findall(r"\\boxed\{([0-9\.,]+)\}", text)
    if boxed:
        return boxed[-1].replace(",", "").strip()

    # 2. Look for explicit answer statements in the response
    patterns = [
        r"(?:the answer is|final answer is|is equal to|equals|is)\s*[:=]?\s*([0-9\.,]+)",
        r"([0-9\.,]+)\s*%",
        r"([0-9\.,]+)\s*(?:feet|foot|inches|meters|\$)"
    ]
    for pattern in patterns:
        match = re.findall(pattern, text, re.IGNORECASE)
        if match:
            return match[-1].replace(",", "").strip()

    # 3. Fallback: Last numeric token
    all_numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if all_numbers:
        return all_numbers[-1].replace(",", "").strip()

    return ""

def parse_model_output(full_output: str):
    """Separates <think>...</think> from the public final answer."""
    thinking_trace = ""
    final_response = full_output

    if "<think>" in full_output and "</think>" in full_output:
        parts = full_output.split("</think>")
        thinking_trace = parts[0].replace("<think>", "").strip()
        final_response = parts[1].strip()
    elif "<think>" in full_output:
        thinking_trace = full_output.replace("<think>", "").strip()
        final_response = ""

    # Clean special tokens
    for token in ["<|im_end|>", "<|endoftext|>"]:
        thinking_trace = thinking_trace.replace(token, "").strip()
        final_response = final_response.replace(token, "").strip()

    return thinking_trace, final_response

def evaluate_monolithic(input_file: str, output_file: str, adapter_path: str, limit: int = 15):
    print("=" * 70)
    print("THESIS STEP: EVALUATING FINE-TUNED MONOLITHIC CONTROL ADAPTER")
    print(f"Adapter: {adapter_path}")
    print("=" * 70)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}! Train it first using train_lora.py.")

    # 1. Load Test Dataset (GSM-Symbolic)
    if not os.path.exists(input_file):
        alt_file = "data/raw/gsm8k_train_5k.json"
        print(f"⚠️ '{input_file}' not found. Falling back to '{alt_file}'...")
        input_file = alt_file

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter unique problem templates
    seen_ids = set()
    unique_subset = []
    for item in data:
        q_id = item.get("id")
        if q_id not in seen_ids:
            seen_ids.add(q_id)
            unique_subset.append(item)
        if len(unique_subset) >= limit:
            break

    print(f"✓ Loaded {len(unique_subset)} unique test problems from {input_file}.\n")

    # 2. Load Base Model + Fine-Tuned Adapter (Unsloth)
    print(f"[1/2] Loading model with adapter: {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    stop_token_ids = [tokenizer.eos_token_id]
    for special in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if isinstance(tid, int):
            stop_token_ids.append(tid)

    results = []
    correct_count = 0

    print("\n[2/2] Running Inference (Greedy, deterministic evaluation)...")
    print("=" * 70)

    for i, item in enumerate(unique_subset):
        q_id = item.get("id", i)
        instance_id = item.get("instance", 0)
        question = item["question"]
        target = str(item.get("target_answer", "")).strip()

        # Prompt with the exact system instruction used during fine-tuning
        system_instruction = "Solve the following math problem by thinking step-by-step inside <think>...</think> tags, then provide the concise final answer."
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = (
                f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            temperature=0.0,    # Deterministic greedy decoding (Page 10)
            do_sample=False
        )
        elapsed = time.time() - start_time

        # Decode newly generated tokens
        generated_tokens = outputs[0][prompt_len:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        # Parse reasoning trace vs final answer
        thinking_trace, final_response = parse_model_output(raw_output)

        # Extract answer from the final response (or fallback to thinking)
        extracted_answer = extract_numeric_answer(final_response)
        if not extracted_answer:
            extracted_answer = extract_numeric_answer(thinking_trace)

        is_correct = (extracted_answer == target) if target else False
        if is_correct:
            correct_count += 1

        print(f"\n[{i+1}/{len(unique_subset)}] Problem ID: {q_id} (Instance: {instance_id})")
        print(f"Time: {elapsed:.2f}s | Target: {target} | Extracted: {extracted_answer} | Match: {'✓' if is_correct else '✗'}")
        print("-" * 50)
        print(f"[Thinking Trace]:\n{thinking_trace}")
        print(f"\n[Final Response]:\n{final_response}")
        print("-" * 50)

        results.append({
            "id": q_id,
            "instance": instance_id,
            "question": question,
            "target_answer": target,
            "thinking_trace": thinking_trace,
            "final_response": final_response,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct
        })

    # Save Results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    em_accuracy = (correct_count / len(unique_subset)) * 100
    print("\n" + "=" * 70)
    print("MONOLITHIC BASELINE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total Evaluated:               {len(unique_subset)}")
    print(f"Exact Match (EM) Accuracy:     {correct_count}/{len(unique_subset)} ({em_accuracy:.1f}%)")
    print(f"Known-Correct Intersection:    {correct_count} problems available for Trap Testing")
    print(f"Saved evaluation results to:   {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Monolithic adapter on GSM-Symbolic.")
    parser.add_argument("--adapter", type=str, default="./finetuned_adapters/adapter_monolithic")
    parser.add_argument("--input", type=str, default="data/raw/gsm_symbolic_test.json")
    parser.add_argument("--output", type=str, default="data/processed/qwen_monolithic_eval.json")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    evaluate_monolithic(
        input_file=args.input,
        output_file=args.output,
        adapter_path=args.adapter,
        limit=args.limit
    )