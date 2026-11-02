"""Generation stages: the base model's own traces, clean evaluation of the three conditions and the Trap Test.

step selects the weights for every stage: 0 is the untrained base model (same prompts, no adapter),
"final" the finished adapter, and an integer N the adapter checkpoint after N training steps.
"""
import pandas as pd

from . import config
from .answers import answers_match, classify_trap_answer, extract_answer, split_thinking
from .data import load_jsonl
from .inference import load_for_inference, run_generation, unload
from .prompts import build_forced_prefix_prompt, build_prompt

SOLVER_FOR_CONDITION = {"soft_modular": "soft_solver", "hard_bottleneck": "hard_solver"}
ROLE_FOR_CONDITION = {"monolithic": "monolithic", **SOLVER_FOR_CONDITION}


def adapter_path(role, step):
    if step == 0:
        return None
    adapter = config.ADAPTER_DIR / f"adapter_{role}"
    return adapter if step == "final" else adapter / "checkpoints" / f"checkpoint-{step}"


def _stage(role, step, jobs, out_path, max_new_tokens, stop_at_think_close=False):
    """Load one adapter (or the base model), generate for all jobs, unload it. Returns {key: record}."""
    path = adapter_path(role, step)
    print(f"\n--- Stage: {role}, step {step} ({path or config.BASE_MODEL}) ---")
    done = {r["key"]: r for r in load_jsonl(out_path)}
    if all(j["key"] in done for j in jobs):
        print(f"{out_path.name}: all {len(jobs)} already generated")
        return done
    model, tokenizer = load_for_inference(path)
    try:
        return run_generation(model, tokenizer, jobs, out_path, max_new_tokens, stop_at_think_close)
    finally:
        unload(model, tokenizer)


# ==========================================
# Own traces (self-distillation data)
# ==========================================
def run_own_traces(problems, out_path):
    """Base Qwen3-8B, thinking mode, no system prompt: its own trace and answer for every problem."""
    jobs = [{"key": p["id"], **p, "prompt": build_prompt("reasoning", question=p["question"])} for p in problems]
    return _stage("reasoning", 0, jobs, out_path, config.MAX_NEW_TOKENS_TRACE)


# ==========================================
# Clean evaluation (full pipelines)
# ==========================================
def run_clean_evaluation(problems, step="final", out_dir=config.EVAL_DIR):
    """Q -> (R, A) for Monolithic; Q -> R then (Q, R) -> A or R -> A for the modular conditions."""
    by_id = {p["id"]: p for p in problems}

    mono = _stage("monolithic", step,
                  [{"key": p["id"], "prompt": build_prompt("monolithic", question=p["question"])} for p in problems],
                  out_dir / f"clean_monolithic_{step}.jsonl", config.MAX_NEW_TOKENS_FULL)

    # One Thinker pass, shared by both modular conditions
    thinker = _stage("thinker", step,
                     [{"key": p["id"], "prompt": build_prompt("thinker", question=p["question"])} for p in problems],
                     out_dir / f"clean_thinker_{step}.jsonl", config.MAX_NEW_TOKENS_TRACE, stop_at_think_close=True)
    # A trace that never closed </think> is unfinished; the pipeline gives no answer for it
    traces = {key: split_thinking(r["generation"])[0] for key, r in thinker.items() if not r["hit_token_limit"]}

    solver_outputs = {}
    for condition, role in SOLVER_FOR_CONDITION.items():
        jobs = [{"key": pid, "prompt": build_prompt(role, question=by_id[pid]["question"], reasoning=trace)}
                for pid, trace in traces.items()]
        solver_outputs[condition] = _stage(role, step, jobs, out_dir / f"clean_{role}_{step}.jsonl",
                                           config.MAX_NEW_TOKENS_ANSWER)

    rows = []
    for p in problems:
        pid = p["id"]
        mono_trace, mono_answer_text = split_thinking(mono[pid]["generation"])
        mono_finished = "</think>" in mono[pid]["generation"]
        per_condition = {
            "monolithic": (mono_trace, mono_answer_text if mono_finished else "", not mono_finished),
            **{c: (split_thinking(thinker[pid]["generation"])[0], out[pid]["generation"] if pid in out else "",
                   thinker[pid]["hit_token_limit"])
               for c, out in solver_outputs.items()},
        }
        for condition, (trace, answer_text, trace_unfinished) in per_condition.items():
            answer = extract_answer(answer_text)
            rows.append({
                "id": pid, "template_id": p["template_id"], "condition": condition, "step": step,
                "target_answer": p["target_answer"], "answer": answer,
                "correct": answers_match(answer, p["target_answer"]),
                "trace_unfinished": trace_unfinished, "trace": trace, "answer_text": answer_text,
            })
    return pd.DataFrame(rows)


# ==========================================
# Trap Test (supplied traces)
# ==========================================
def _trap_prompt(condition, question, trace):
    if condition == "monolithic":
        return build_forced_prefix_prompt(question, trace)
    return build_prompt(SOLVER_FOR_CONDITION[condition], question=question, reasoning=trace)


def run_trap_test(traps, step="final", out_dir=config.TRAP_DIR):
    """Every condition answers from the same clean and trapped traces, in its own input format."""
    outputs = {}
    for condition, role in ROLE_FOR_CONDITION.items():
        jobs = []
        for t in traps:
            for variant, trace in (("clean", t["clean_trace"]), ("trap", t["trapped_trace"])):
                jobs.append({"key": f"{t['id']}|{variant}", "prompt": _trap_prompt(condition, t["question"], trace)})
        outputs[condition] = _stage(role, step, jobs, out_dir / f"trap_test_{condition}_{step}.jsonl",
                                    config.MAX_NEW_TOKENS_ANSWER)

    rows = []
    for t in traps:
        for condition, out in outputs.items():
            clean_answer = extract_answer(out[f"{t['id']}|clean"]["generation"])
            trap_answer_text = out[f"{t['id']}|trap"]["generation"]
            answer = extract_answer(trap_answer_text)
            rows.append({
                "id": t["id"], "template_id": t["template_id"], "condition": condition, "step": step,
                "correct_answer": t["target_answer"], "trap_answer": t["trap_answer"],
                "answer_clean_trace": clean_answer, "answer": answer,
                "clean_outcome": classify_trap_answer(clean_answer, t["target_answer"], t["trap_answer"]),
                "outcome": classify_trap_answer(answer, t["target_answer"], t["trap_answer"]),
                "answer_changed": not answers_match(answer, clean_answer),
                "answer_tokens": out[f"{t['id']}|trap"]["generated_tokens"],
                "answer_text": trap_answer_text,
            })
    return pd.DataFrame(rows)
