"""Single source of truth for paths, model names and hyperparameters.

Every notebook imports from here so the three conditions are guaranteed to
share the same settings. Values are sized for one RTX 4070 (12GB VRAM).
"""
from pathlib import Path

# ==========================================
# Paths
# ==========================================
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # GSM8K / GSM-Symbolic pulls
TRACE_DIR = DATA_DIR / "traces"         # Base Qwen3-8B's own traces and answers (self-distillation)
SFT_DIR = DATA_DIR / "sft"              # Training sets, one file per role
EVAL_DIR = DATA_DIR / "eval"            # Clean evaluation generations
TRAP_DIR = DATA_DIR / "trap"            # Trapped traces, reviews, trap-test outputs
RESULTS_DIR = ROOT / "results"          # Final tables and statistics

ADAPTER_DIR = ROOT / "finetuned_adapters"

GSM8K_TRAIN_PATH = RAW_DIR / "gsm8k_train.json"
GSM_SYMBOLIC_PATH = RAW_DIR / "gsm_symbolic.json"
GSM8K_VALID_PATH = TRACE_DIR / "gsm8k_valid.json"        # Training pool
SYMBOLIC_VALID_PATH = TRACE_DIR / "symbolic_valid.json"  # Trap pool

# ==========================================
# Models
# ==========================================
# Every stage (trace generation, training, evaluation) uses this 4-bit checkpoint
BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

# The teacher only writes trapped traces; the training data is the base model's own
TEACHER_MODEL = "deepseek-v4-flash"     # DeepSeek-V4.1-Flash; set to the API name in use
TEACHER_BASE_URL = "https://api.deepseek.com"
TEACHER_WORKERS = 5

# ==========================================
# Data preparation
# ==========================================
SEED = 3407

# GSM-Symbolic has 100 templates x 50 instances. A seeded sample of this many instances per template
# (15 -> 1,500 questions) is used for the clean evaluation and the Trap Test; None uses all 5,000.
SYMBOLIC_PER_TEMPLATE = 15

# Thinking budget for the base model and the Thinker. A trace that has not closed </think> by
# then is treated as unfinished (in practice almost always a reasoning loop).
MAX_NEW_TOKENS_TRACE = 4096
# A trace that repeats the same sentence (over LOOP_MIN_CHARS characters) LOOP_MIN_REPEATS
# times is treated as looping, even if it finished
LOOP_MIN_REPEATS = 3
LOOP_MIN_CHARS = 25

MAX_SEQ_LENGTH = 4096                   # Longest full training example (prompt + completion)

# Traps are generated for a template-balanced subset of the problems the base model solves
TRAP_CANDIDATES = 1500
# The Trap Test reports the first TRAP_TEST_SIZE traps in the known-correct intersection that
# pass the automatic checks and are approved by hand. Every trap that passes the automatic
# checks is also tested, as a larger secondary set.
TRAP_TEST_SIZE = 100

# ==========================================
# QLoRA (identical for every adapter)
# ==========================================
LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

TRAIN_CONFIG = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,       # Effective batch size 8
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="adamw_8bit",
    logging_steps=5,
    save_steps=25,                       # Adapter-only checkpoints, kept for the checkpoint analysis
    save_total_limit=None,
    save_only_model=True,
    seed=SEED,
)

# Checkpoints evaluated in the Trap Test. 0 = the untrained base model with the same prompts.
CHECKPOINT_STEPS = [0, 25, 50, 100, 200, 400, "final"]

# ==========================================
# Generation (deterministic everywhere)
# ==========================================
GEN_BATCH_SIZE = 4                       # Left-padded batches; lower if the 4070 runs out of memory
MAX_NEW_TOKENS_ANSWER = 768              # Solvers and forced-prefix Monolithic answers (free-form)
MAX_NEW_TOKENS_FULL = MAX_NEW_TOKENS_TRACE + MAX_NEW_TOKENS_ANSWER  # Monolithic: trace and answer
MAX_INFERENCE_LENGTH = 6144              # Prompt + generation; the longest is a Soft Solver input
