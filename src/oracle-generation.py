import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

RAW_DATA_PATH = "data/raw/gsm8k_train_5k.json"
PROCESSED_DIR = "data/processed"
MAX_TRACE_TOKENS = 1500  # Thesis length constraint for 12GB VRAM

def extract_numeric_value(text):
    """Extract numeric value from answer text (handles \boxed{}, decimals, and plain numbers)."""
    if not text:
        return None
    # Check for \boxed{val}
    boxed_match = re.search(r"\\boxed\{([0-9\.,]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).replace(",", "").strip()
    
    # Fallback: extract last number in the text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None

def estimate_tokens(text):
    """Rough token estimation (~1.3 tokens per word)."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def process_single_problem(client, sample):
    """Call DeepSeek API (V4 Flash) for a single GSM8K problem."""
    question = sample["question"]
    target_numeric = sample["target_answer"]

    system_prompt = (
        "You are an expert mathematical reasoner. Solve the problem step by step.\n\n"
        "Do NOT repeat, paraphrase, or quote the question or input problem in your thinking process.\n"\
        "At the very end of your response, restate the final answer with: 'The final answer is \\boxed{number}'."
    )

    try:
        # Updated payload structure for DeepSeek V4 Flash
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        message = response.choices[0].message
        
        # Extract thought trace and final text
        reasoning_trace = getattr(message, "reasoning_content", "") or ""
        final_answer_text = message.content or ""

        # --- FILTRATION PIPELINE (Page 11) ---
        # 1. Length Filter: Check token length of trace
        trace_token_len = estimate_tokens(reasoning_trace)
        if trace_token_len > MAX_TRACE_TOKENS or trace_token_len == 0:
            return None, "Length_Filtered"

        # 2. Validation Filter: Verify numeric match with GSM8K target
        extracted_numeric = extract_numeric_value(final_answer_text)
        if not extracted_numeric or str(extracted_numeric) != str(target_numeric):
            return None, "Validation_Filtered"

        # Validated item
        return {
            "id": sample["id"],
            "question": question,
            "reasoning_trace": reasoning_trace,
            "final_answer": final_answer_text,
            "target_answer": target_numeric
        }, "Success"

    except Exception as e:
        return None, f"API_Error: {e}"
    
def generate_synthetic_data(limit=None, max_workers=5):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file!")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)

    if limit:
        print(f"Limit flag set: Processing first {limit} problems for testing...")
        gsm8k_data = gsm8k_data[:limit]

    print(f"\nStarting generation for {len(gsm8k_data)} problems using DeepSeek-V3.2 API...")

    validated_results = []
    stats = {"Success": 0, "Length_Filtered": 0, "Validation_Filtered": 0, "API_Error": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_problem, client, item): item for item in gsm8k_data}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Distilling Reasoning"):
            result, status = future.result()
            if status.startswith("API_Error"):
                stats["API_Error"] += 1
            else:
                stats[status] += 1
            
            if result:
                validated_results.append(result)

    print("\n" + "=" * 60)
    print("DISTILLATION & FILTRATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Processed:       {len(gsm8k_data)}")
    print(f"✓ Validated Samples:   {stats['Success']}")
    print(f"✗ Failed Validation:   {stats['Validation_Filtered']}")
    print(f"✗ Exceeded Token Limit:{stats['Length_Filtered']}")
    print(f"✗ API Errors:          {stats['API_Error']}")

    # --- FORMAT & SAVE 3 DATASETS (Page 11) ---
    save_formatted_datasets(validated_results)

def save_formatted_datasets(results):
    print("\nCreating 3 JSON fine-tuning datasets...")

    monolithic_set = []
    thinker_set = []
    solver_set = []

    for item in results:
        q = item["question"]
        cot = item["reasoning_trace"]
        ans = item["final_answer"]

        # 1. Baseline Monolithic Set
        monolithic_set.append({
            "instruction": "Solve the following math problem by thinking step-by-step inside <think>...</think> tags, then provide the concise final answer.",
            "input": q,
            "output": f"<think>\n{cot}\n</think>\n{ans}"
        })

        # 2. Thinker Only Set
        thinker_set.append({
            "instruction": "Generate a step-by-step reasoning trace to solve the following math problem.",
            "input": q,
            "output": cot
        })

        # 3. Solver Only Set (Matches Page 14: question + "\n" + reasoning_trace)
        solver_set.append({
            "instruction": "Based on the provided reasoning trace, output the concise final answer.",
            "input": f"{cot}",
            "output": ans
        })

    # Save to disk
    paths = {
        "baseline_monolithic.json": monolithic_set,
        "thinker_only.json": thinker_set,
        "solver_only.json": solver_set
    }

    for filename, data in paths.items():
        out_path = os.path.join(PROCESSED_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(data)} items to {out_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data from DeepSeek API.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems to process (for testing).")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent API request workers.")
    args = parser.parse_args()

    generate_synthetic_data(limit=args.limit, max_workers=args.workers)