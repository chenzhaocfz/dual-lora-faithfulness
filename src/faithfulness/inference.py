"""Deterministic, resumable batched generation for each adapter role.

Stage isolation in the modular conditions:
  * Stages run one after another over the whole question set. The Thinker adapter is loaded,
    generates every trace, and is removed from GPU memory before a Solver adapter is loaded.
  * Every model.generate() call builds a new KV cache from its own input_ids; no
    past_key_values are ever passed between calls, so nothing from the Thinker's
    (question-conditioned) cache can reach a Solver.
  * The Solver's input is rebuilt from text by prompts.build_prompt, so the Hard Bottleneck
    Solver's input contains only R.
"""
# Unsloth must be imported before transformers / peft
from unsloth import FastLanguageModel

import gc

import torch
from tqdm.auto import tqdm

from . import config
from .data import append_jsonl, load_jsonl
from .prompts import IM_END, THINK_CLOSE


def load_for_inference(adapter_path=None):
    """Load an adapter directory (or checkpoint) on the 4-bit base; None loads the untrained base model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path or config.BASE_MODEL),
        max_seq_length=config.MAX_INFERENCE_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    return model, tokenizer


def unload(model, tokenizer=None):
    """Drop a model (and its adapter) from GPU memory between stages."""
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def stop_token_ids(tokenizer, stop_at_think_close=False):
    ids = [tokenizer.convert_tokens_to_ids(IM_END)]
    if stop_at_think_close:
        ids.append(tokenizer.convert_tokens_to_ids(THINK_CLOSE))
    return ids


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens, stop_at_think_close=False):
    """Greedy decoding (no sampling). Returns decoded continuations with special tokens kept."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=stop_token_ids(tokenizer, stop_at_think_close),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,  # a fresh cache per call; never reused across calls
    )
    continuations = outputs[:, inputs.input_ids.shape[1]:]
    texts = tokenizer.batch_decode(continuations, skip_special_tokens=False)
    return [t.replace(tokenizer.pad_token, "") for t in texts]


def run_generation(model, tokenizer, jobs, out_path, max_new_tokens, stop_at_think_close=False,
                   batch_size=config.GEN_BATCH_SIZE):
    """Generate for every job {"key", "prompt", ...} not already in out_path (JSONL, resumable).

    Jobs are sorted by prompt length so batches need little padding; the order is fixed,
    so re-running gives the same batches.
    """
    done = {r["key"] for r in load_jsonl(out_path)}
    todo = sorted((j for j in jobs if j["key"] not in done), key=lambda j: (len(j["prompt"]), j["key"]))
    print(f"{out_path.name}: {len(done)} done, {len(todo)} to generate")

    for i in tqdm(range(0, len(todo), batch_size), desc=out_path.stem):
        batch = todo[i:i + batch_size]
        texts = generate_batch(model, tokenizer, [j["prompt"] for j in batch], max_new_tokens, stop_at_think_close)
        stops = (IM_END, THINK_CLOSE) if stop_at_think_close else (IM_END,)
        for job, text in zip(batch, texts):
            n_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
            # A generation that ends without a stop token ran out of max_new_tokens
            append_jsonl({**job, "generation": text, "generated_tokens": n_tokens,
                          "hit_token_limit": not text.rstrip().endswith(stops)}, out_path)

    return {r["key"]: r for r in load_jsonl(out_path)}
