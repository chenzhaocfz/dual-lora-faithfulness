# Investigating Reasoning-Trace Dependence in LLMs via Modular LoRA Information Bottlenecks

**Author:** Chen Zhao  
**Institution:** Sheridan College  
**Supervision:** Dr. Ghassem Tofighi  
**Base Model:** `unsloth/Qwen3-8B-unsloth-bnb-4bit`  
**Trap Generator:** DeepSeek-V4.1-Flash (reasoning mode)  
**Compute:** 1× NVIDIA GeForce RTX 4070 (12GB VRAM)  
**Frameworks:** Unsloth, Hugging Face Transformers/PEFT, PyTorch  

---

## 1. Research Questions

In this study, a quantitative experimental approach is used to ask whether the information available to a Large
Language Model (LLM) during answer generation affects its use of a reasoning trace.

The study considers three conditions built on the Qwen3-8B model: a Monolithic Model, a Soft Modular Model, and a
Hard Bottleneck Model. Each model is trained using reasoning traces from GSM8K problems. They are then tested on
GSM-Symbolic problems, where the mathematical correctness of each answer is assessed. The study then introduces
errors into valid reasoning traces and checks whether each model follows the modified reasoning in its answer.

By doing this, the study aims to answer two questions about reasoning-trace dependence:

1. Does it matter whether reasoning and answer generation are separate stages of the system?
2. Does it matter if the original question is not available to the model when it generates its answer?

---

## 2. Experimental Design

In all three conditions, Qwen3-8B is used as the base LLM. All three start from the same base model and use the
same source training data. What differs across the conditions is the relationship between
reasoning and answer generation.

```text
Monolithic        Q ──▶ [ Monolithic adapter ] ──▶ R, A
                        (one continuous generation)

Soft Modular      Q ──▶ [ Thinker ] ──▶ R
                  ───── Thinker unloaded, KV cache cleared ─────
                  Q, R ──▶ [ Soft Solver ] ──▶ A

Hard Bottleneck   Q ──▶ [ Thinker ] ──▶ R
                  ───── Thinker unloaded, KV cache cleared ─────
                     R ──▶ [ Hard Solver ] ──▶ A
```

In the Monolithic condition, the model is given the question and generates the reasoning trace and answer in one
continuous output.

In the Soft Modular condition, reasoning and answer generation are split. First, a Thinker adapter takes the
question and generates a reasoning trace. Generation then stops, the Thinker adapter is unloaded, and the
key-value (KV) cache is cleared. Next, the Soft Modular Solver adapter takes the question and reasoning trace and
generates the answer.

In the Hard Bottleneck condition, the question is dropped from the answer stage. The Thinker generates its
reasoning trace from the question, but the Hard Bottleneck Solver sees only the reasoning. All the information from
the question needed by the Solver to produce the answer must be found in the reasoning trace.

The two modular conditions use the same Thinker adapter. Therefore, the study can explore the impact of having (or
not having) the question in the Solver stage separately from the impact of using a modular system architecture.

---

## 3. Pipeline

The work is split into four notebooks, run in order. Shared code lives in `src/faithfulness/`, so training and
evaluation always use the same prompts, answer parsing and settings.

| Notebook | GPU | What it does |
| :--- | :---: | :--- |
| `00_tests.ipynb` | ✓ | Environment, offline unit tests, teacher API call, 4-bit load, deterministic decoding, stage isolation, forced prefix, training memory on 12GB |
| `01_datasets.ipynb` | ✓ | GSM8K and GSM-Symbolic, the base model's own traces, filtering, one training set per adapter, trap candidates |
| `02_training.ipynb` | ✓ | The Monolithic, Thinker, Soft Modular Solver and Hard Bottleneck Solver adapters |
| `03_evaluation.ipynb` | ✓ | Clean accuracy, known-correct intersection, manual check, Trap Test across checkpoints, analysis |

```text
           GSM8K train (7,473)             GSM-Symbolic (1,500 of 5,000)
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
          base Qwen3-8B (4-bit), greedy, thinking mode: own (R, A)
                                      │
                                      ▼
        keep if finished within 4,096 tokens, A correct (+ not looping: GSM8K)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          GSM8K: training pool               GSM-Symbolic: trap pool
                    │                                   │
                    ▼                                   ▼
       train Monolithic, Thinker,            template-balanced order
        Soft Solver, Hard Solver           DeepSeek: R becomes R_trap
       (checkpoint every 25 steps)              automatic checks
                    │                                   │
                    ▼                                   │
         GSM-Symbolic: clean EM                         │
                    │                                   │
                    ▼                                   │
       known-correct intersection  ◀────────────────────┘
                    │
                    ▼
       first 100 approved by hand
                    │
                    ▼
  Trap Test on R and R_trap, every condition and checkpoint
                    │
                    ▼
     TD, Cochran's Q, McNemar + Holm
```

### 3.1 Model Preparation: Self-Distillation
In early tests, the Qwen3-8B model tended to get stuck in reasoning loops under deterministic decoding (see
`00_tests`, section 5). Rather than training a separate reasoning adapter to avoid this, the study uses the base
model's own reasoning and filters out the loops. The base model answers every GSM8K training question and every
sampled GSM-Symbolic question once, greedily, in thinking mode and without a system prompt. A generation is kept only if:

* it closes its reasoning (`</think>`) and finishes its answer within 4,096 new tokens
* the answer matches the correct answer
* training questions only: the reasoning does not repeat the same sentence three or more times (a loop that
  happened to finish), so the adapters do not learn to loop

All three conditions therefore start from the same untrained base model, and every adapter learns only its role
from the base model's own traces and answers. Training on a teacher's traces would also teach the adapters a new
reasoning and answering style: in a pilot with teacher traces and one-sentence answers, every condition followed
every trap, which leaves no room to compare conditions.

### 3.2 Training Data Preparation
The GSM8K questions that pass the filter form the training pool. Each gives a sample

$$(Q, R, A),$$

where $Q$ is the question from GSM8K, $R$ is the base model's reasoning trace, and $A$ is the answer it wrote after
its reasoning (ending in `\boxed{}`). From these samples, one training set is built for each adapter:

| Adapter | Input | Target |
| :--- | :--- | :--- |
| Monolithic | $Q$ | $(R, A)$ |
| Thinker | $Q$ | $R$ |
| Soft Modular Solver | $(Q, R)$ | $A$ |
| Hard Bottleneck Solver | $R$ | $A$ |

All four sets use the same samples: a sample is dropped from every set if its longest training example (Monolithic
or Soft Solver, which carry $Q$, $R$ and $A$) is longer than 4,096 tokens. Both Solver training sets include the
same reasoning trace and answer. They differ in that the question is dropped from the input for the Hard Bottleneck
Solver.

Loss is applied to the target tokens only. The prompt text for each adapter is in `src/faithfulness/prompts.py`.
Qwen3 thinking mode is on for the adapters that write $R$, and off (an empty `<think></think>` block) for the Solvers.
The Solver prompts do not instruct the model to trust the reasoning trace.

The GSM-Symbolic questions that pass the filter form the trap pool (4.3): every trap starts from a trace with which
the base model reached the correct answer.

### 3.3 Fine-Tuning
The models are fine-tuned with 4-bit Quantized Low-Rank Adaptation (QLoRA) to work within the 12GB VRAM limit. Each
adapter is trained directly on the 4-bit Qwen3-8B base model, under the same configuration:

* LoRA rank 16, LoRA alpha 16, dropout 0, applied to the attention and feed-forward projection layers
  (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
* Batch size 2 with gradient accumulation 4 (effective batch size 8), learning rate $2\times10^{-4}$, cosine
  schedule with 5% warm-up, 8-bit AdamW, 1 epoch, seed 3407
* Maximum sequence length 4,096 tokens
* An adapter-only checkpoint every 25 steps, so trace dependence can be measured during training

For each adapter, training steps, loss, model checkpoints, runtime, peak VRAM and the number of training tokens are
recorded in `finetuned_adapters/adapter_<role>/training_summary.json`.

### 3.4 Inference Procedure
In the Monolithic condition, the model is presented with each question and produces its reasoning and its answer in
one pass.

In the modular conditions, the Thinker adapter is first presented with the question and produces its reasoning.
Generation is then halted, the Thinker adapter is unloaded, the KV cache is cleared, and the Solver stage begins.
The Soft Modular Solver is presented with $(Q, R)$, while the Hard Bottleneck Solver is presented with only $R$.

In practice, the stages run one after another over the whole question set: the Thinker generates every trace and
is unloaded before a Solver adapter is loaded. Each generation builds a new KV cache from its own input, and the
Solver input is rebuilt from text, so no cached state from the question can reach the Solver. `00_tests` checks
that a Solver gives identical output with or without a preceding Thinker call.

Generation settings are the same across conditions:

* Deterministic decoding (no sampling), on the same 4-bit base model used for training
* At most 4,096 new tokens of reasoning (the Thinker, and the base model's own traces); 4,096 + 768 for the
  Monolithic model's reasoning and answer; 768 for the Solvers and forced-prefix answers
* Prompts sorted by length and generated in fixed batches (`GEN_BATCH_SIZE`), so re-running gives the same batches

---

## 4. Evaluation

Deterministic decoding is used across all conditions, so that differences between conditions do not result from
sampling. The assessment has two parts: the mathematical performance of each model, and how much the final answer
depends on a supplied reasoning trace.

### 4.1 Clean Mathematical Accuracy
All three models are tested on GSM-Symbolic (`apple/GSM-Symbolic`, `main`). For each problem, the answer produced
by each full pipeline is compared with the answer in the benchmark:

$$EM = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}}$$

GSM-Symbolic has 100 templates with 50 instances each. A seeded random sample of 15 instances per template
(1,500 questions) is used, which keeps every template represented while fitting the 4070's time budget
(`SYMBOLIC_PER_TEMPLATE`; set it to `None` for all 5,000). A Thinker trace that does not
finish within the token limit counts as a wrong answer for both modular conditions. As a baseline, the same three
pipelines are also run on the untrained base model with the same prompts.

### 4.2 Known-Correct Intersection
The Trap Test is only applied to problems that all three conditions solve correctly before any intervention.
Otherwise, a wrong answer following an intervention could reflect a flaw in the mathematical ability of the model,
rather than its use of the reasoning chain.

Trapped traces are written before training, for a seeded, template-balanced selection of the trap pool (4.3).
After the clean evaluation, the traps whose problem all three conditions solve are kept in that selection order.
The first 100 in this order that pass the automatic and manual checks are used for the main Trap Test. The selection
is recorded in `data/trap/trap_selection.json`, and the number of problems available at each step is reported.

### 4.3 Adversarial Trap Test
For each problem, a valid reasoning trace is taken and a deliberate arithmetic or logical error introduced (for
example, $7 \times 8 = 56$ changed to $7 \times 8 = 54$). Later steps are rewritten to be consistent with this
error, giving a trace with a different expected answer.

* The valid trace is the base model's own reasoning trace for the problem (3.1), with which it reached the
  correct answer.
* The trapped traces are generated by DeepSeek-V4.1-Flash for the problems in the trap pool (up to 1,500), in a
  seeded, template-balanced order (one instance per template, then a second, and so on).
* Automatic checks are run first: the implied answer differs from the correct one, the trace was changed and ends
  on the trap answer, the changed step can be found, no value given in the problem was changed, and the trace
  contains no note about the error.
* Each trace used in the main Trap Test is then checked by hand in `data/trap/trap_review.csv`: it includes an
  error, later steps follow from that error, and the implied answer differs from the original correct answer. A
  rejected trace is replaced by the next one in the selection order.
* Every trap in the known-correct intersection that passes the automatic checks is also tested, as a larger
  secondary set without the manual check.

Each condition is fed the same reasoning trace, with the inputs available to its answer stage kept as defined in
the experimental design:

* Monolithic: the question and trapped reasoning trace are provided as a forced prefix
  ($Q$ + `<think>` $R_{trap}$ `</think>`), leaving the model to generate only the answer
* Soft Modular: $(Q, R_{trap}) \rightarrow A$
* Hard Bottleneck: $R_{trap} \rightarrow A$

Each condition is also tested using the corresponding unmodified trace in the same input format.

### 4.4 Trace-Dependence Measure
The model's answer is compared with the answer suggested by its trapped reasoning trace. If they match, the model
is said to have "followed" the trap:

$$TD = \frac{\text{Number of Trap Follows}}{\text{Number of Trap Tests}}$$

The study also records whether the answer changes relative to the corresponding unmodified trace. Answers that are
neither the original correct answer nor the trap answer are counted separately as "Other Errors", and are not
counted as following the trap. Following a trapped trace is treated as evidence of answer–trace agreement, not
proof that the written trace fully represents the model's internal computation.

### 4.5 Analysis
Each condition sees the same set of problems, so the results are paired. Cochran's Q test is used first to see
whether the three conditions differ on the binary trap outcome (follow or not follow). If it does, McNemar tests
compare Monolithic vs. Soft Modular, Soft Modular vs. Hard Bottleneck, and Monolithic vs. Hard Bottleneck, with
Holm correction across the three tests.

The main analysis uses the final adapters on the manually checked set. Trace dependence is also reported at
training checkpoints (steps 0, 25, 50, 100, 200, 400 and final, where step 0 is the untrained base model with the
same prompts), and the same tests are repeated on the secondary set for the untrained and final models.

---

## 5. Repository Structure

```text
dual-lora-faithfulness/
├── readme.md
├── requirements.txt
├── .env.example                      # DEEPSEEK_API_KEY
├── notebooks/
│   ├── 00_tests.ipynb
│   ├── 01_datasets.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/faithfulness/
│   ├── config.py                     # Paths, model names, every hyperparameter
│   ├── prompts.py                    # Prompt format for each adapter role (train = inference)
│   ├── answers.py                    # Answer extraction and trap classification
│   ├── data.py                       # Benchmarks, file I/O, own-trace filtering, DeepSeek trap generation
│   ├── training.py                   # QLoRA training
│   ├── inference.py                  # Deterministic, resumable batched generation
│   ├── evaluation.py                 # Own traces, the three conditions: clean evaluation and Trap Test
│   └── analysis.py                   # EM, TD, Cochran's Q, McNemar, Holm
├── data/                             # (git-ignored) created by the notebooks
│   ├── raw/                          # GSM8K, GSM-Symbolic
│   ├── traces/                       # The base model's own traces, filtered pools, filter stats
│   ├── sft/                          # One training set per role + dataset_summary.json
│   ├── eval/                         # Clean-evaluation generations (JSONL) and results
│   └── trap/                         # Selection, trapped traces, manual check, Trap Test outputs
├── finetuned_adapters/               # (git-ignored) adapter_<role>/ + checkpoints/ + training_summary.json
└── results/                          # Final tables: training report, EM, TD, statistical tests
```

---

## 6. Setup

Target environment: Ubuntu 22.04 / WSL2, Python 3.11, CUDA 12.x, RTX 4070 (12GB).

```bash
git clone https://github.com/chenzhaocfz/dual-lora-faithfulness.git
cd dual-lora-faithfulness

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then add your DeepSeek API key
jupyter lab notebooks/
```

All settings (model names, token limits, loop filter, trap counts, LoRA and training hyperparameters, checkpoints,
generation batch size)
are in `src/faithfulness/config.py`. Set `TEACHER_MODEL` to the API model name of DeepSeek-V4.1-Flash.

### Running on 12GB
* Run the notebooks top to bottom. In `02_training`, run one training cell at a time, and restart the kernel if
  memory is not released between adapters.
* `00_tests` section 8 trains a few steps at the full 4,096-token length and reports peak VRAM before any real run.
* The base model's own traces (`01_datasets`, section 2) are the longest step: about 9,000 generations of up to
  4,096 tokens, which takes days on a 4070. The clean evaluation of the 1,500 GSM-Symbolic questions takes about a day.
* Trace generation, trap generation, training and every evaluation stage save progress as they go, so they can be
  interrupted and resumed.
* If generation runs out of memory, lower `GEN_BATCH_SIZE`.
* Use `PILOT_LIMIT` (`01_datasets`) and `DRY_RUN_STEPS` (`02_training`) for a quick end-to-end check first.
