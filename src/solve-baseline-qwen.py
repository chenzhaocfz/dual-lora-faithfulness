import os
import json
import re
import argparse
import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from unsloth import FastLanguageModel

class StopAtEndOfThink(StoppingCriteria):
    """Halts generation as soon as the model finishes its thinking block </think>."""
    def __init__(self, tokenizer, prompt_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=False)
        return "</think>" in generated_text

def extract_boxed_or_last_number(text: str) -> str:
    """Extracts numeric answer inside \\boxed{}, or the last stated number."""
    if not text:
        return ""
    # Look for \boxed{...} anywhere in the trace
    boxed = re.findall(r"\\boxed\{([0-9\.,]+)\}", text)
    if boxed:
        return boxed[-1].replace(",", "").strip()

    # Look for explicit conclusion patterns near the end of thinking
    tail = text[-300:]
    patterns = [
        r"(?:the answer is|final answer is|equals|is)\s*[:=]?\s*([0-9\.,]+)",
        r"([0-9\.,]+)\s*%",
    ]
    for pattern in patterns:
        match = re.findall(pattern, tail, re.IGNORECASE)
        if match:
            return match[-1].replace(",", "").strip()

    all_nums = re.findall(r"[-+]?\d*\.?\d+", tail)
    return all_nums[-1].replace(",", "").strip() if all_nums else ""

def solve_baseline_subset(input_file: str, output_file: str, limit: int = 15, max_tokens: int = 1536):
    print("=" * 70)
    print("THESIS STEP: EXTRACTING NATURAL <think> REASONING TRACES")
    print("=" * 70)

    if not os.path.exists(input_file):
        alt_file = "data/raw/gsm8k_train_5k.json"
        print(f"⚠️ '{input_file}' not found. Falling back to '{alt_file}'...")
        input_file = alt_file

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Unique template IDs
    seen_ids = set()
    unique_subset = []
    for item in data:
        q_id = item.get("id")
        if q_id not in seen_ids:
            seen_ids.add(q_id)
            unique_subset.append(item)
        if len(unique_subset) >= limit:
            break

    print(f"✓ Selected {len(unique_subset)} unique question templates.\n")

    MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    correct_count = 0

    print("\n" + "=" * 70)
    print(f"GENERATING TRACES (max_new_tokens={max_tokens}, Auto-stop at </think>)")
    print("=" * 70)

    for i, item in enumerate(unique_subset):
        q_id = item.get("id", i)
        instance_id = item.get("instance", 0)
        question = item["question"]
        target = str(item.get("target_answer", "")).strip()

        # Let the model naturally invoke its reasoning template
        messages = [{"role": "user", "content": question}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        # Stop immediately when </think> is completed
        stop_criteria = StoppingCriteriaList([StopAtEndOfThink(tokenizer, prompt_len)])

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            stopping_criteria=stop_criteria,
            use_cache=True,
            temperature=0.0,
            do_sample=False
        )
        elapsed = time.time() - start_time

        raw_output = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)

        # Isolate the thought trace inside <think>...</think>
        if "<think>" in raw_output and "</think>" in raw_output:
            thinking_trace = raw_output.split("<think>")[1].split("</think>")[0].strip()
        else:
            thinking_trace = raw_output.replace("<think>", "").replace("</think>", "").strip()

        extracted_answer = extract_boxed_or_last_number(thinking_trace)
        is_correct = (extracted_answer == target) if target else False
        if is_correct:
            correct_count += 1

        print(f"\n[{i+1}/{len(unique_subset)}] ID: {q_id} (Instance: {instance_id})")
        print(f"Elapsed: {elapsed:.2f}s | Target: {target} | Qwen: {extracted_answer} | Match: {'✓' if is_correct else '✗'}")
        print("-" * 50)
        # Display the last 300 chars of the thought trace to see how it concluded
        print("Trace Conclusion:")
        print("..." + thinking_trace[-300:] if len(thinking_trace) > 300 else thinking_trace)

        results.append({
            "id": q_id,
            "instance": instance_id,
            "question": question,
            "ground_truth_cot": item.get("ground_truth_cot", ""),
            "target_answer": target,
            "qwen_generated_thinking": thinking_trace,
            "qwen_extracted_answer": extracted_answer,
            "is_correct": is_correct
        })

    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total Unique Problems:        {len(unique_subset)}")
    print(f"Known-Correct (Intersection):   {correct_count}/{len(unique_subset)} ({correct_count/len(unique_subset)*100:.1f}%)")
    print(f"Traces saved to:              {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clean reasoning traces up to </think>.")
    parser.add_argument("--input", type=str, default="data/raw/gsm_symbolic_test.json")
    parser.add_argument("--output", type=str, default="data/processed/qwen_baseline_traces.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=1536)
    args = parser.parse_args()

    solve_baseline_subset(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        max_tokens=args.max_tokens
    )