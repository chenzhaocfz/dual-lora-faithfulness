"""Answer parsing shared by teacher filtering, clean evaluation and the Trap Test."""
import re

from .prompts import IM_END, THINK_CLOSE, THINK_OPEN

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def _boxed_contents(text):
    """Contents of every \\boxed{...}, handling nested braces such as \\boxed{\\frac{1}{2}}."""
    contents = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, start = 1, match.end()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[start:i])
                    break
    return contents


def normalize_number(value):
    """'1,200.00' / '$1200' / '1200.' -> '1200'. Returns '' if no number is present."""
    if value is None:
        return ""
    numbers = re.findall(_NUMBER, str(value).replace("$", ""))
    if not numbers:
        return ""
    raw = numbers[-1].replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return ""
    if number == int(number):
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def extract_answer(text):
    """Final numeric answer: last \\boxed{}, else 'final answer is X', else the last number."""
    if not text:
        return ""
    boxed = _boxed_contents(text)
    if boxed:
        return normalize_number(boxed[-1])

    stated = re.findall(rf"(?:final answer|the answer)\s*(?:is|:|=)\s*\$?\s*({_NUMBER})", text, re.IGNORECASE)
    if stated:
        return normalize_number(stated[-1])

    return normalize_number(text)


def answers_match(a, b):
    a, b = normalize_number(a), normalize_number(b)
    return a != "" and a == b


def gsm_target(raw_answer):
    """GSM8K / GSM-Symbolic store the gold answer after '####'."""
    return normalize_number(raw_answer.split("####")[-1]) if "####" in raw_answer else ""


def split_thinking(text):
    """Split a generation into (trace, answer) around </think>, dropping special tokens."""
    text = text.replace(IM_END, "").replace("<|endoftext|>", "")
    if THINK_CLOSE in text:
        trace, answer = text.split(THINK_CLOSE, 1)
    else:
        trace, answer = text, ""
    return trace.replace(THINK_OPEN, "").strip(), answer.strip()


def classify_trap_answer(answer, correct_answer, trap_answer):
    """'follow' (matches the trapped trace), 'original' (matches the clean answer) or 'other'."""
    if answers_match(answer, trap_answer):
        return "follow"
    if answers_match(answer, correct_answer):
        return "original"
    return "other"
