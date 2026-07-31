import os
import json
from datasets import load_dataset

DATA_RAW_DIR = "data/raw"

def prepare_datasets():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    print("=" * 60)
    print("THESIS DATASET PREPARATION")
    print("=" * 60)

    # 1. Load GSM8K (Training split for DeepSeek Distillation)
    print("\n[1/2] Loading GSM8K (openai/gsm8k)...")
    gsm8k_dataset = load_dataset("gsm8k", "main", split="train")
    
    # Select 5,000 problems as specified in thesis (Page 11)
    gsm8k_5k = gsm8k_dataset.select(range(min(5000, len(gsm8k_dataset))))
    
    gsm8k_data_list = []
    for idx, sample in enumerate(gsm8k_5k):
        # Extract question and official numeric target answer
        question = sample["question"]
        raw_answer = sample["answer"]
        # GSM8K standard format ends answer with '#### <numeric_val>'
        final_numeric = raw_answer.split("####")[-1].strip() if "####" in raw_answer else ""
        
        gsm8k_data_list.append({
            "id": idx,
            "question": question,
            "ground_truth_cot": raw_answer,
            "target_answer": final_numeric
        })

    gsm8k_out_path = os.path.join(DATA_RAW_DIR, "gsm8k_train_5k.json")
    with open(gsm8k_out_path, "w", encoding="utf-8") as f:
        json.dump(gsm8k_data_list, f, indent=2)
    print(f"✓ Saved 5,000 GSM8K problems to: {gsm8k_out_path}")

    # 2. Load GSM-Symbolic (Apple Benchmark for Evaluation)
    print("\n[2/2] Loading GSM-Symbolic (apple/GSM-Symbolic)...")
    try:
        gsm_symbolic_ds = load_dataset("apple/GSM-Symbolic", name="main", split="test")
        gsm_symbolic_list = []
        
        for sample in gsm_symbolic_ds:
            question = sample["question"]
            raw_answer = sample["answer"]
            final_numeric = raw_answer.split("####")[-1].strip() if "####" in raw_answer else ""
            
            gsm_symbolic_list.append({
                "id": sample.get("id", len(gsm_symbolic_list)),
                "instance": sample.get("instance", 0),
                "question": question,
                "ground_truth_cot": raw_answer,
                "target_answer": final_numeric
            })

        symbolic_out_path = os.path.join(DATA_RAW_DIR, "gsm_symbolic_test.json")
        with open(symbolic_out_path, "w", encoding="utf-8") as f:
            json.dump(gsm_symbolic_list, f, indent=2)
        print(f"✓ Saved {len(gsm_symbolic_list)} GSM-Symbolic evaluation samples to: {symbolic_out_path}")
    except Exception as e:
        print(f"⚠️ Could not load apple/GSM-Symbolic directly: {e}")
        print("Fallback: Using standard GSM8K test split for validation...")

    # 3. Print Inspection Sample
    print("\n" + "=" * 60)
    print("SAMPLE DATASET INSPECTION (GSM8K Sample #1)")
    print("=" * 60)
    sample_item = gsm8k_data_list[0]
    print(f"QUESTION:\n{sample_item['question']}\n")
    print(f"GROUND TRUTH ANSWER:\n{sample_item['ground_truth_cot']}\n")
    print(f"EXTRACTED NUMERIC TARGET: {sample_item['target_answer']}")

if __name__ == "__main__":
    prepare_datasets()