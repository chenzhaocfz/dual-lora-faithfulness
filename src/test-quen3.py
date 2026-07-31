import torch
import time
from unsloth import FastLanguageModel

def test_qwen3_loading():
    print("=" * 60)
    print("THESIS TEST SCRIPT: Qwen-3-8B 4-bit Initialization & LoRA Setup")
    print("=" * 60)

    MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    
    MAX_SEQ_LENGTH = 2048
    DTYPE = None  # Auto-detection (float16 / bfloat16 for Ampere architecture RTX 4070)
    LOAD_IN_4BIT = True

    print(f"\n[1/4] Loading model: {MODEL_NAME}...")
    start_time = time.time()

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        print(f"✓ Model loaded successfully in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        print(f"✗ Failed to load {MODEL_NAME}: {e}")
        print("Tip: If the repo name differs slightly on HuggingFace, try 'unsloth/Qwen3-8B-Base-unsloth-bnb-4bit' or 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'.")
        return

    # LORA
    print("\n[2/4] Applying 4-bit QLoRA target modules...")
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
    )
    print("LoRA adapters injected successfully")

    # 3. Test Inference Pass
    print("\n[3/4] Testing dummy inference generation...")
    FastLanguageModel.for_inference(model) # Enable fast 2x faster inference mode
    
    prompt = "Question:The area of Carlos's rectangular living room is 630 square feet. If the length of his room is 7 yards, what is the perimeter of the room in feet?"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        use_cache=True,
        temperature=0.0, # Greedy decoding as specified in thesis (Page 10)
        do_sample=False
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("-" * 40)
    print("Sample Output Generation:")
    print(response)
    print("-" * 40)

    # 4. Measure VRAM Consumption for RTX 4070 (12GB)
    print("\n[4/4] Hardware & VRAM Usage Check:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        max_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {allocated:.2f} GB / {max_vram:.2f} GB")
        print(f"VRAM Reserved:  {reserved:.2f} GB / {max_vram:.2f} GB")
        
        if reserved <= 10.0:
            print("\nPASSED: Memory usage is within the RTX 4070 12GB ceiling!")
            print(f"Headroom available for KV Cache & Dual-LoRA adapters: ~{max_vram - reserved:.2f} GB")
        else:
            print("\nWARNING: High memory usage. Lower max_seq_length if needed.")

if __name__ == "__main__":
    test_qwen3_loading()