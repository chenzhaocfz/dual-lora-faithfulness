# Investigating Reasoning-Trace Dependence in LLMs via Modular LoRA Information Bottlenecks

**Author:** Chen Zhao  
**Institution:** Sheridan College  
**Supervision:** Dr. Ghassem Tofighi  
**Base Model:** `unsloth/Qwen3-8B-unsloth-bnb-4bit`  
**Compute Constraints:** 1× NVIDIA GeForce RTX 4070 (12GB VRAM)  
**Primary Frameworks:** Unsloth, Hugging Face TRL (`SFTTrainer`), PyTorch  

---

## 1. Abstract & Research Questions

Chain-of-Thought (CoT) prompting frequently exhibits **reasoning unfaithfulness**: models often generate post-hoc explanations that rationalize an answer rather than deriving it step-by-step. In such cases, if intermediate reasoning errors are introduced, models frequently bypass the corrupted steps and output the original, uncorrupted answer.

This thesis investigates whether restricting the model's access to the question context during answer synthesis enforces strict behavioral dependence on the reasoning trace. Specifically, we evaluate two core questions:
1. **Architectural Separation:** Does physically isolating reasoning generation from answer synthesis across modular stages (with explicit Key-Value cache purging) increase trace dependence?
2. **Information Bottlenecking:** Does stripping direct question access from the answer stage ($R \rightarrow A$) prevent question-to-answer leakage and force the model to rely strictly on the intermediate reasoning trace?

---

## 2. Experimental Architecture

The study implements a 3-way comparative design. All experimental conditions are initialized from the same base checkpoint (`unsloth/Qwen3-8B-unsloth-bnb-4bit`) and trained on the same distilled GSM8K reasoning distribution:

```text
Condition 1: Monolithic Baseline
[ Question Q ] ──────────────────────────► [ <think> Reasoning R </think> + Answer A ]
(Single continuous generation pass, shared attention context)

Condition 2: Soft Modular Dual-LoRA
[ Question Q ] ──► [ Thinker Adapter ] ──► [ Reasoning R ]
                                                  │
             [ KV-CACHE PURGED & RESET ]          │
                                                  ▼
[ Question Q ] ──────────────────────────► [ Soft Solver Adapter ] ──► [ Answer A ]
(Two-stage execution; Question Q reinjected into Solver input)

Condition 3: Hard Bottleneck Dual-LoRA
[ Question Q ] ──► [ Thinker Adapter ] ──► [ Reasoning R ]
                                                  │
             [ KV-CACHE PURGED & RESET ]          │
                                                  ▼
                                           [ Hard Solver Adapter ] ──► [ Answer A ]
(Two-stage execution; Question Q completely omitted from Solver stage)
```

### Architectural Comparison

| Condition | Stage 1 (Reasoning) | Stage 2 (Answer) | KV-Cache Policy | Question at Answer Stage |
| :--- | :--- | :--- | :--- | :--- |
| **Monolithic Baseline** | $Q \rightarrow (R, A)$ | N/A (Single Pass) | Persistent | **Yes** (Direct context) |
| **Soft Modular Model** | $Q \rightarrow R$ | $(Q, R) \rightarrow A$ | **Purged & Reset** | **Yes** (Reinjected) |
| **Hard Bottleneck Model**| $Q \rightarrow R$ | $R \rightarrow A$ | **Purged & Reset** | **No** (Strict bottleneck) |

Both modular conditions share the exact same **Thinker** adapter head, isolating the effect of question presence in Stage 2 from the modular split itself.

---

## 3. Methodological Pipeline

```text
  [ Raw GSM8K (5k) ]
          │
          ▼
  [ DeepSeek Flash API Distillation ]
  (reasoning_effort="high", extra_body={"thinking": {"type": "enabled"}})
          │
          ▼
  [ Filtration Pipeline ]
  ├── 1. Length Filter: Trace ≤ 1500 tokens
  └── 2. Validation: Extracted answer == Ground Truth (exact match)
          │
          ▼
  [ Fine-Tuning 3 LoRA Adapters via Unsloth ] (r=16, alpha=16)
  ├── baseline_monolithic.json   ──► adapter_monolithic
  ├── thinker_only.json          ──► adapter_thinker
  └── solver_only.json           ──► adapter_solver
          │
          ▼
  [ Evaluation on GSM-Symbolic ]
  ├── 1. Clean Exact Match (EM) Accuracy
  ├── 2. Known-Correct Intersection (Target N ≤ 100)
  └── 3. Adversarial Trap Test (Prefix-forced fault injection)
```

### 3.1 Dataset Distillation & Filtration
* **Source:** 5,000 problems from the GSM8K training split (`openai/gsm8k`).
* **Oracle:** `deepseek-flash` via the official DeepSeek API. Reasoning traces are extracted natively via `message.reasoning_content` with system prompts requiring `\boxed{}` formatted answers.
* **Length Filtering:** Reasoning chains exceeding 1,500 tokens are discarded to ensure compatibility with our 12GB VRAM training ceiling.
* **Validation Filtering:** Distilled samples are retained only if the teacher model's final numeric output exactly matches the ground-truth target.

### 3.2 Fine-Tuning Setup
* **Quantization:** 4-bit NormalFloat (NF4) via Unsloth.
* **LoRA Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **Hyperparameters:** Rank $r=16$, Alpha $\alpha=16$, Dropout $=0$, Batch Size $=2$, Gradient Accumulation $=4$ (effective batch size 8), Learning Rate $=2\times 10^{-4}$ with Cosine schedule.
* **Data Format:** Conversational ChatML with intermediate reasoning encapsulated within `<think>...</think>` tokens.

### 3.3 Inference & Stopping Criteria
* **Greedy Deterministic Decoding:** All evaluations run at `temperature = 0.0` with `do_sample = False`.
* **Custom Stopping Criteria:** The pipeline implements `StopAtEndOfThink(StoppingCriteria)` to halt generation during the Thinker stage the moment the closing `</think>` token is decoded, saving computation and isolating the trace.

---

## 4. Evaluation Framework & Metrics

### 4.1 Clean Mathematical Accuracy
Evaluated on **GSM-Symbolic** (`apple/GSM-Symbolic`) to measure out-of-distribution mathematical generalization under unmodified conditions:

$$\text{EM} = \frac{\text{Correct Predictions}}{\text{Total Test Samples}}$$

### 4.2 Known-Correct Intersection
To prevent confounding reasoning unfaithfulness with baseline arithmetic failure, the Adversarial Trap Test is strictly evaluated on problems that **all three model configurations solve correctly** prior to intervention.

### 4.3 Adversarial Trap Test & Trace Dependence ($TD$)
For each problem in the intersection:
1. A valid reasoning trace is perturbed by introducing an arithmetic or logical error (e.g., $7 \times 8 = 56 \to 54$).
2. Downstream derivations are rewritten to maintain internal consistency with the injected error, leading to an altered final answer.
3. The perturbed trace $R_{\text{trap}}$ is provided as input:
   * **Monolithic:** Prefix-forced context $[Q, R_{\text{trap}}]$.
   * **Soft Modular:** Input $(Q, R_{\text{trap}})$ to the solver after cache reset.
   * **Hard Bottleneck:** Input $R_{\text{trap}}$ only to the solver after cache reset.

Trace dependence measures how often the model follows the corrupted reasoning chain rather than overriding it:

$$TD = \frac{\text{Number of Trap-Consistent Answers}}{\text{Total Interventions}}$$

* **Statistical Significance:** Paired comparisons across conditions are evaluated using **Cochran's $Q$ test**, followed by post-hoc pairwise **McNemar tests** with **Holm-Bonferroni correction**.

---

## 5. Repository Structure

```text
dual-lora-faithfulness/
├── .env.example
├── requirements.txt
├── README.md
├── build_thesis_notebooks.py        # Generates all .ipynb files from code
│
├── notebooks/                       # Interactive thesis notebooks
│   ├── 01_dataset_preparation.ipynb # Ingests GSM8K 5k & GSM-Symbolic
│   ├── 02_oracle_distillation.ipynb # Multithreaded DeepSeek distillation
│   ├── 03_test_qwen3_and_vram.ipynb # 4-bit Unsloth & VRAM diagnostics
│   ├── 04_train_lora_adapters.ipynb # SFTTrainer for Monolithic, Thinker, Solver
│   └── 05_evaluation_and_traces.ipynb # Trace extraction & GSM-Symbolic evaluation
│
├── src/                             # Core Python modules
│   ├── load_datasets.py             # Dataset loaders and normalizers
│   ├── oracle_generation.py         # DeepSeek API concurrent distiller
│   ├── train_lora.py                # Unsloth QLoRA training script
│   ├── evaluate_monolithic.py       # Deterministic evaluation harness
│   └── solve_baseline_subset.py     # StopAtEndOfThink trace extractor
│
├── data/
│   ├── raw/                         # GSM8K 5k & GSM-Symbolic raw JSONs
│   └── processed/                   # SFT datasets & evaluation traces
│
└── finetuned_adapters/              # Saved LoRA checkpoints
    ├── adapter_monolithic/
    ├── adapter_thinker/
    └── adapter_solver/
```

---

## 6. Installation & Quickstart

### 6.1 Environment Setup
Tested on Ubuntu 22.04 / WSL2 with CUDA 12.1+ and Python 3.10.

```bash
git clone https://github.com/chenzhaocfz/dual-lora-faithfulness.git
cd dual-lora-faithfulness

python -m venv venv
source venv/bin/activate

# Install dependencies (Unsloth + PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets openai tqdm python-dotenv
```

### 6.2 Configuration
Create a `.env` file in the root directory:
```bash
DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### 6.3 Building the Notebooks
To compile the standalone scripts into clean, valid `.ipynb` files:
```bash
python build_thesis_notebooks.py
```

---

## 7. Execution Guide

You can run the full pipeline either through the interactive notebooks in `notebooks/` or via the command line:

### Step 1: Prepare Datasets
Downloads GSM8K (5k subset) and GSM-Symbolic:
```bash
python src/load_datasets.py
```

### Step 2: Distill Reasoning Traces
Calls the DeepSeek Flash API to generate validated reasoning traces:
```bash
# Run pilot test on 20 samples
python src/oracle_generation.py --limit 20 --workers 5

# Run full distillation
python src/oracle_generation.py --workers 5
```

### Step 3: Verify Hardware & VRAM Footprint
Ensures the 4-bit base model and LoRA parameters fit comfortably inside the 12GB VRAM budget:
```bash
python src/test_qwen3.py
```

### Step 4: Fine-Tune Adapters
Trains the three experimental adapters:
```bash
# Train individual adapters
python src/train_lora.py --adapter monolithic --epochs 1 --lr 2e-4
python src/train_lora.py --adapter thinker --epochs 1 --lr 2e-4
python src/train_lora.py --adapter solver --epochs 1 --lr 2e-4

# Or train all sequentially
python src/train_lora.py --adapter all
```

### Step 5: Evaluate Monolithic Baseline
Evaluates Exact Match accuracy on GSM-Symbolic:
```bash
python src/evaluate_monolithic.py \
    --adapter ./finetuned_adapters/adapter_monolithic \
    --input data/raw/gsm_symbolic_test.json \
    --output data/processed/qwen_monolithic_eval.json \
    --limit 100
```