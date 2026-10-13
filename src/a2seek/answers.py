"""Parse the original list, JSON and Qwen grounding answer formats safely."""

import ast
import json
import math
import re


def tag_content(text, tag):
    if not isinstance(text, str):
        return None
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def valid_box(box):
    shape = (
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in box)
    )
    if not shape:
        return False
    try:
        return all(math.isfinite(x) for x in box) and box[2] >= box[0] and box[3] >= box[1]
    except OverflowError:
        return False


def parse_answer(text):
    if not isinstance(text, str) or len(text) > 100_000:
        return None, None
    answer = tag_content(text, "answer")
    text = answer if answer is not None else text.strip()
    try:
        refs = re.findall(r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>", text, re.DOTALL)
        raw_boxes = re.findall(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
        if refs:
            categories = [s.strip() for s in refs]
            boxes = []
            for raw in raw_boxes:
                value = ast.literal_eval(raw.strip())
                if len(value) == 2:
                    value = [*value[0], *value[1]]
                boxes.append(list(value))
        else:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            if ";" in text:
                cats, raw_boxes = text.split(";", 1)
                categories, boxes = ast.literal_eval(cats.strip()), ast.literal_eval(raw_boxes.strip())
            else:
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    value = ast.literal_eval(text)
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    categories = [x["category"] for x in value]
                    # Keep box positions aligned with their categories when an object has no box.
                    boxes = [x.get("bbox", [0, 0, 0, 0]) for x in value]
                else:
                    categories, boxes = value, []
        if not isinstance(categories, (list, tuple)) or not categories or not all(isinstance(c, str) and c for c in categories):
            return None, None
        if boxes is None:
            boxes = []
        if not isinstance(boxes, (list, tuple)) or not all(valid_box(b) for b in boxes):
            return None, None
        return list(categories), [list(b) for b in boxes]
    except (ValueError, SyntaxError, TypeError, KeyError, IndexError, RecursionError):
        return None, None


def iou(left, right):
    if not valid_box(left) or not valid_box(right):
        return 0.0
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    overlap = width * height
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - overlap
    try:
        finite = math.isfinite(overlap) and math.isfinite(union)
    except OverflowError:
        finite = False
    if not finite or (
        union == 0 and left[2] > left[0] and left[3] > left[1] and right[2] > right[0] and right[3] > right[1]
    ):
        # Scale only extreme inputs; preserve the arithmetic of ordinary pixel coordinates.
        scale = max(abs(x) for x in [*left, *right])
        if scale == 0:
            return 0.0
        left = [x / scale for x in left]
        right = [x / scale for x in right]
        width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
        height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
        overlap = width * height
        union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - overlap
    return overlap / union if union > 0 else 0.0
