"""Prompt formats for every adapter role.

Training and inference both build their text from this module, so the prompt an
adapter sees at evaluation time is byte-identical to the one it was trained on.

Qwen3 ChatML is written out by hand (rather than via apply_chat_template) so the
thinking-mode setting of each role is explicit:

    role          user message      assistant prefix (prompt)   completion (trained / generated)
    reasoning     Q                 <think>\n                   R \n</think>\n\n A <|im_end|>   (base model's own traces; not trained)
    monolithic    Q                 <think>\n                   R \n</think>\n\n A <|im_end|>
    thinker       Q                 <think>\n                   R \n</think> <|im_end|>
    soft_solver   Q + R             <think>\n\n</think>\n\n     A <|im_end|>
    hard_solver   R                 <think>\n\n</think>\n\n     A <|im_end|>

Thinking mode is ON for roles that write R and OFF (empty think block, Qwen3's
non-thinking convention) for the Solvers, which only write A.
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
EMPTY_THINK = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

SYSTEM_PROMPTS = {
    # Self-distillation: the base model's own traces, no system prompt, Qwen3's native thinking format
    "reasoning": None,
    "monolithic": (
        "Solve the math problem. Reason step by step, then give the final answer "
        "in the form: The final answer is \\boxed{number}."
    ),
    "thinker": "Reason step by step about how to solve the math problem.",
    # Solver prompts are deliberately neutral: they never tell the model to trust the trace
    "soft_solver": (
        "You are given a math problem and a reasoning trace about it. Give the final answer "
        "to the problem in the form: The final answer is \\boxed{number}."
    ),
    "hard_solver": (
        "You are given a reasoning trace about a math problem. Give the final answer "
        "to the problem in the form: The final answer is \\boxed{number}."
    ),
}

THINKING_ROLES = {"reasoning", "monolithic", "thinker"}
SOLVER_ROLES = {"soft_solver", "hard_solver"}


def _user_content(role, question, reasoning):
    if role == "soft_solver":
        return f"Problem:\n{question.strip()}\n\nReasoning trace:\n{reasoning.strip()}"
    if role == "hard_solver":
        return f"Reasoning trace:\n{reasoning.strip()}"
    return question.strip()


def build_prompt(role, question=None, reasoning=None):
    """Everything the model is conditioned on, ending at the point where generation starts."""
    if role not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown role: {role}")
    if role == "hard_solver":
        question = None  # The Hard Bottleneck Solver never sees Q, even if a caller passes it

    text = ""
    if SYSTEM_PROMPTS[role]:
        text += f"{IM_START}system\n{SYSTEM_PROMPTS[role]}{IM_END}\n"
    text += f"{IM_START}user\n{_user_content(role, question, reasoning)}{IM_END}\n"
    text += f"{IM_START}assistant\n"
    text += f"{THINK_OPEN}\n" if role in THINKING_ROLES else EMPTY_THINK
    return text


def build_completion(role, reasoning=None, answer=None):
    """Training target for a role (the tokens that receive loss)."""
    if role in ("reasoning", "monolithic"):
        return f"{reasoning.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}{IM_END}"
    if role == "thinker":
        return f"{reasoning.strip()}\n{THINK_CLOSE}{IM_END}"
    if role in SOLVER_ROLES:
        return f"{answer.strip()}{IM_END}"
    raise ValueError(f"Unknown role: {role}")


def build_forced_prefix_prompt(question, reasoning):
    """Monolithic Trap Test input: Q plus a supplied trace forced into the think block.

    The model only generates what follows </think>, i.e. the answer.
    """
    return build_prompt("monolithic", question=question) + f"{reasoning.strip()}\n{THINK_CLOSE}\n\n"


def build_example(role, question, reasoning, answer):
    """A (prompt, completion) pair for supervised fine-tuning."""
    return {
        "prompt": build_prompt(role, question=question, reasoning=reasoning),
        "completion": build_completion(role, reasoning=reasoning, answer=answer),
    }
