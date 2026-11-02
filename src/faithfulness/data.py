"""Dataset loading, file I/O, own-trace filtering and the DeepSeek teacher (trap generation)."""
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from . import config
from .answers import answers_match, extract_answer, gsm_target, split_thinking
from .prompts import THINK_CLOSE, build_example


# ==========================================
# File I/O
# ==========================================
def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ==========================================
# Benchmarks
# ==========================================
def load_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [
        {"id": f"gsm8k-{i}", "question": row["question"], "target_answer": gsm_target(row["answer"])}
        for i, row in enumerate(ds)
    ]


def load_gsm_symbolic():
    """GSM-Symbolic 'main' (100 templates x 50 instances). Each item keeps its template id."""
    from datasets import load_dataset

    ds = load_dataset("apple/GSM-Symbolic", name="main", split="test")
    return [
        {
            "id": f"sym-{row['id']}-{row['instance']}",
            "template_id": row["id"],
            "instance": row["instance"],
            "question": row["question"],
            "target_answer": gsm_target(row["answer"]),
        }
        for row in ds
    ]



def sample_per_template(problems, k=config.SYMBOLIC_PER_TEMPLATE, seed=config.SEED, round_size=5):
    """A seeded random sample of k instances from every template (all problems if k is None).

    Instances are drawn in rounds of round_size per template (round r uses seed + r), so a larger
    sample always contains every smaller one: 15 per template = the 5-per-template sample + 10 more.
    """
    if k is None:
        return problems
    remaining = {}
    for p in problems:
        remaining.setdefault(p["template_id"], []).append(p)
    sample = []
    for r in range(-(-k // round_size)):
        rng = random.Random(seed + r)
        n = min(round_size, k - r * round_size)
        for t in sorted(remaining):
            drawn = rng.sample(remaining[t], n)
            sample += drawn
            remaining[t] = [p for p in remaining[t] if p not in drawn]
    return sorted(sample, key=lambda p: (p["template_id"], p["instance"]))

# ==========================================
# Own traces (self-distillation)
# ==========================================
def is_looping(trace, min_repeats=config.LOOP_MIN_REPEATS, min_chars=config.LOOP_MIN_CHARS):
    """True if some sentence longer than min_chars appears min_repeats or more times."""
    sentences = [s.strip() for s in re.split(r"[.\n?!]", trace) if len(s.strip()) > min_chars]
    return any(n >= min_repeats for n in Counter(sentences).values())


def filter_own_traces(records, check_loops=True):
    """Keep the base model's generations that finished, do not loop and reach the gold answer.

    records: run_generation output for build_prompt("reasoning", Q). Returns (kept, stats).
    check_loops: drop traces that repeat a sentence (training pool only; for the trap pool a correct,
    finished trace is enough, and repeated re-checking is normal in it).
    """
    stats = Counter()
    kept = []
    for r in records:
        if r["hit_token_limit"] or THINK_CLOSE not in r["generation"]:
            stats["unfinished (no </think> within the token limit)"] += 1
            continue
        trace, answer = split_thinking(r["generation"])
        if check_loops and is_looping(trace):
            stats["looping"] += 1
            continue
        if not answers_match(extract_answer(answer), r["target_answer"]):
            stats["wrong answer"] += 1
            continue
        stats["kept"] += 1
        kept.append({"id": r["id"], "template_id": r.get("template_id"), "question": r["question"],
                     "target_answer": r["target_answer"], "trace": trace, "answer": answer,
                     "generated_tokens": r["generated_tokens"]})
    kept.sort(key=lambda r: (r["id"].split("-")[0], [int(x) for x in r["id"].split("-")[1:]]))
    return kept, {"total": len(records), **stats}


def build_training_sets(valid, tokenizer, max_seq_length=config.MAX_SEQ_LENGTH):
    """One training set per experimental role, all from the same samples.

    A sample is kept only if its longest example (Monolithic and Soft Solver carry Q, R and A) fits
    in max_seq_length, so every role is trained on exactly the same questions, traces and answers.
    """
    def n_tokens(ex):
        return len(tokenizer(ex["prompt"] + ex["completion"], add_special_tokens=False).input_ids)

    keep = [v for v in valid
            if max(n_tokens(build_example(role, v["question"], v["trace"], v["answer"]))
                   for role in ("monolithic", "soft_solver")) <= max_seq_length]
    sets = {role: [{"id": v["id"], **build_example(role, v["question"], v["trace"], v["answer"])} for v in keep]
            for role in ("monolithic", "thinker", "soft_solver", "hard_solver")}
    return sets, {"valid": len(valid), "too_long": len(valid) - len(keep), "kept": len(keep)}


def select_trap_candidates(valid, n=config.TRAP_CANDIDATES, seed=config.SEED):
    """Template-balanced order: templates are shuffled, then one instance is taken from each in turn."""
    by_template = {}
    for v in valid:
        by_template.setdefault(v["template_id"], []).append(v["id"])
    rng = random.Random(seed)
    templates = sorted(by_template)
    rng.shuffle(templates)
    for t in templates:
        by_template[t].sort()
        rng.shuffle(by_template[t])
    ordered = []
    for round_ in range(max(len(ids) for ids in by_template.values())):
        ordered += [by_template[t][round_] for t in templates if round_ < len(by_template[t])]
    return ordered[:n]


# ==========================================
# DeepSeek teacher
# ==========================================
def teacher_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(config.ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(f"DEEPSEEK_API_KEY not found in the environment or {config.ROOT / '.env'}")
    return OpenAI(api_key=api_key, base_url=config.TEACHER_BASE_URL)


def ask_teacher(client, system_prompt, user_prompt, json_mode=False):
    """One reasoning-mode call. Returns (reasoning_content, content)."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=config.TEACHER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        **kwargs,
    )
    message = response.choices[0].message
    return getattr(message, "reasoning_content", "") or "", message.content or ""


# ==========================================
# Trap generation
# ==========================================
TRAP_SYSTEM_PROMPT = """You create controlled errors in math reasoning traces for a research experiment.

You will receive a math problem and a correct reasoning trace for it. Rewrite the trace so that:
1. Exactly ONE arithmetic or logical step is changed to be wrong (for example "7 * 8 = 56" becomes "7 * 8 = 54").
2. Every later step uses the wrong value consistently, so the trace confidently reaches a different final answer.
3. Everything else (wording, style, structure, earlier steps) stays the same as the original.
4. The trace never mentions, hints at or corrects the error.
5. The new final answer must differ from the original answer.
6. Never change a number that is given in the problem, including where the trace restates or quotes the
   problem. The error must be in a step the reasoning works out, not in the problem's data.

Reply with a JSON object with these keys:
  "original_step": the exact text of the step you changed, copied from the original trace
  "modified_step": the exact text of that step after your change
  "trapped_trace": the full rewritten trace
  "trapped_answer": the final numeric answer the rewritten trace arrives at (number only)"""


def make_trap(client, item):
    """One trapped trace from the teacher, as a record for auto_check_trap. API errors are kept in "error"."""
    user_prompt = (
        f"Problem:\n{item['question']}\n\n"
        f"Original reasoning trace:\n{item['clean_trace']}\n\n"
        f"Original final answer: {item['target_answer']}"
    )
    try:
        _, content = ask_teacher(client, TRAP_SYSTEM_PROMPT, user_prompt, json_mode=True)
        trap = json.loads(content)
        return {**item,
                "original_step": trap.get("original_step", ""),
                "modified_step": trap.get("modified_step", ""),
                "trapped_trace": trap.get("trapped_trace", "").strip(),
                "trap_answer": extract_answer(str(trap.get("trapped_answer", ""))),
                "error": None}
    except Exception as e:
        return {**item, "error": str(e)}


def run_trap_generation(items, out_path, workers=config.TEACHER_WORKERS):
    """Generate one trapped trace per item (resumable, like the distillation run)."""
    done = {r["id"] for r in load_jsonl(out_path) if not r["error"]}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} traps already generated, {len(todo)} to go")
    if not todo:
        return

    client = teacher_client()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(make_trap, client, it) for it in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating traps"):
            append_jsonl(future.result(), out_path)


def auto_check_trap(trap):
    """Automatic checks run before manual review. Returns a list of problems (empty = passed)."""
    if trap.get("error"):
        return [f"api error: {trap['error']}"]
    problems = []
    if not trap["trapped_trace"]:
        problems.append("empty trapped trace")
    if trap["trapped_trace"].strip() == trap["clean_trace"].strip():
        problems.append("trace unchanged")
    if not trap["trap_answer"]:
        problems.append("no trap answer")
    elif answers_match(trap["trap_answer"], trap["target_answer"]):
        problems.append("trap answer equals correct answer")
    elif not answers_match(extract_answer(trap["trapped_trace"][-400:]), trap["trap_answer"]):
        problems.append("trapped trace does not end on the trap answer")
    # The teacher sometimes appends a note such as "Original final answer: 42", which gives the trap away
    if re.search(r"original final answer|trapped trace|injected error", trap["trapped_trace"], re.IGNORECASE):
        problems.append("teacher note left in trapped trace")
    # Answers like 62.8169 or 21.428571 get rounded by the models, so a real follow would not match exactly
    if re.search(r"\.\d{3,}", trap["trap_answer"] or ""):
        problems.append("trap answer has 3+ decimal places")
    # A given value from the problem that the clean trace uses but the trapped trace no longer contains
    # means the trap changed the problem's data instead of a reasoning step
    number = r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d.])"
    numbers = lambda text: {n.replace(",", "") for n in re.findall(number, text)}
    lost = (numbers(trap["question"]) & numbers(trap["clean_trace"])) - numbers(trap["trapped_trace"])
    if lost:
        problems.append("changes a given value from the problem")
    if trap["original_step"] and trap["original_step"] not in trap["clean_trace"]:
        problems.append("original_step not found in clean trace")
    if trap["modified_step"] and trap["modified_step"] not in trap["trapped_trace"]:
        problems.append("modified_step not found in trapped trace")
    return problems
