"""Framework-independent scoring and training supervision for A2Seek GRPO."""

import ast
import json
import math
import re

from a2seek.answers import iou, parse_answer, tag_content, valid_box


def parse_seeking(text):
    """Return (required, xyxy); malformed or missing supervision returns (None, None)."""
    if not isinstance(text, str) or len(text) > 100_000:
        return None, None
    content = tag_content(text, "seeking")
    if content is None:
        return None, None
    if content.strip().lower() in {"<no need>", "no need", "none", "false"}:
        return False, None
    match = re.fullmatch(r"<\|box_start\|>(.*?)<\|box_end\|>", content, re.DOTALL)
    try:
        box = ast.literal_eval(match[1].strip() if match else content)
        if isinstance(box, (list, tuple)) and len(box) == 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in box):
            box = [*box[0], *box[1]]
        if valid_box(box) and box[2] > box[0] and box[3] > box[1]:
            return True, list(box)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        pass
    return None, None


def training_solution(row):
    """Build private reward targets before GRPO replaces the assistant message."""
    categories, boxes = parse_answer(row.get("solution"))
    if categories is None:
        objects = row.get("objects") or {}
        categories = objects.get("ref") or [row["anomaly_name"]]
        boxes = objects.get("bbox") or []
    target = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant"), "")
    required, region = parse_seeking(target)
    # Targets may describe the observation request in prose. Resolve
    # its explicit xyxy region, or the accompanying object ROI, during prep.
    content = tag_content(target, "seeking")
    if required is None and content:
        for candidate in re.findall(r"\[[^\[\]]+\]", content):
            required, region = parse_seeking(f"<seeking>{candidate}</seeking>")
            if required is True:
                break
        if required is None:
            candidate = (row.get("objects") or {}).get("region")
            if valid_box(candidate) and candidate[2] > candidate[0] and candidate[3] > candidate[1]:
                required, region = True, list(candidate)
    return json.dumps({"categories": categories, "boxes": boxes,
                       "seeking_required": required, "seeking_box": region},
                      ensure_ascii=False, separators=(",", ":"))


def ground_truth(solution):
    if isinstance(solution, str):
        try:
            value = json.loads(solution)
        except (ValueError, TypeError):
            value = None
        if isinstance(value, dict):
            categories, boxes = parse_answer(json.dumps(value.get("categories")) + ";" + json.dumps(value.get("boxes", [])))
            required, region = value.get("seeking_required"), value.get("seeking_box")
            if required is not None and not isinstance(required, bool):
                required = None
            if required is True and (not valid_box(region) or region[2] <= region[0] or region[3] <= region[1]):
                required, region = None, None
            return categories, boxes, required, region
    categories, boxes = parse_answer(solution)
    return categories, boxes, None, None


def format_score(completion):
    if not isinstance(completion, str):
        return 0.0
    pattern = r"\s*<think>.*?</think>\s*<seeking>.*?</seeking>\s*<answer>.*?</answer>\s*"
    if not re.fullmatch(pattern, completion, re.DOTALL):
        return 0.0
    categories, _ = parse_answer(completion)
    required, _ = parse_seeking(completion)
    return float(categories is not None and required is not None)


def class_score(completion, solution):
    prediction, _ = parse_answer(completion)
    categories, _, _, _ = ground_truth(solution)
    if prediction is None or categories is None:
        return 0.0
    if prediction == categories:
        return 1.0
    return 0.1 if "Normal" not in prediction and "Normal" not in categories else 0.0


def box_score(completion, solution):
    _, predicted = parse_answer(completion)
    categories, reference, _, _ = ground_truth(solution)
    if not predicted or not reference or categories == ["Normal"]:
        return 0.0
    return sum(iou(p, g) for p, g in zip(predicted, reference)) / max(len(predicted), len(reference))


def seeking_score(completion, solution):
    predicted, box = parse_seeking(completion)
    _, _, required, reference = ground_truth(solution)
    if predicted is None or required is None or predicted != required:
        return 0.0
    return iou(box, reference) if required else 1.0


def length_score(completion, solution, tokenizer, max_completion_length):
    think = tag_content(completion, "think")
    answer = tag_content(completion, "answer")
    predicted, _ = parse_answer(completion)
    categories, _, _, _ = ground_truth(solution)
    if not think or answer is None or predicted is None or categories is None:
        return 0.0
    count = len(tokenizer.encode(think, add_special_tokens=False))
    if count == 0:
        return 0.0
    if predicted == categories:
        return min(1.0, 1.0 / math.log1p(count))
    return min(count / max_completion_length, 1.0)
