"""Evaluate prediction JSONL files using explicit, reproducible metrics."""

import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import re

from a2seek.answers import iou, parse_answer, valid_box
from a2seek.data import read_jsonl, sample_id


def align_references(rows, references):
    """Join by sample ID, require full coverage and reject altered annotation targets."""
    rows = list(rows)
    if references is None:
        return rows
    predictions, annotations = {}, {}
    for row in rows:
        key = sample_id(row)
        if key in predictions:
            raise ValueError(f"Duplicate prediction sample: {key}")
        predictions[key] = row
    for row in references:
        key = sample_id(row)
        if key in annotations:
            raise ValueError(f"Duplicate reference sample: {key}")
        annotations[key] = row
    missing, extra = set(annotations) - set(predictions), set(predictions) - set(annotations)
    if missing or extra:
        raise ValueError(f"Prediction coverage mismatch: {len(missing)} missing, {len(extra)} unexpected samples")
    aligned = []
    for key, reference in annotations.items():
        prediction = predictions[key]
        for field in ("code", "anomaly_name", "objects"):
            if field in prediction and prediction[field] != reference.get(field):
                raise ValueError(f"Prediction annotation differs from reference: {key}, field {field}")
        aligned.append(dict(reference, **prediction))
    return aligned


def validate_aliases(aliases):
    if aliases is not None and (not isinstance(aliases, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in aliases.items()
    )):
        raise ValueError("Label aliases must be a JSON object mapping nonempty strings to nonempty strings")
    return aliases or {}


def evaluate(rows, aliases=None, references=None):
    aliases = validate_aliases(aliases)
    groups = defaultdict(list)
    seen = set()
    for row in align_references(rows, references):
        key = sample_id(row)
        if key in seen:
            raise ValueError(f"Duplicate prediction sample: {key}")
        seen.add(key)
        if "prediction" not in row:
            raise ValueError(f"Missing prediction for {key}")
        categories, boxes = parse_answer(row["prediction"])
        categories = [aliases.get(c, c) for c in categories] if categories else None
        objects = row.get("objects") or {}
        if not isinstance(objects, dict) or not isinstance(row["anomaly_name"], str) or not row["anomaly_name"]:
            raise ValueError(f"Invalid reference annotation: {key}")
        gt_name = aliases.get(row["anomaly_name"], row["anomaly_name"])
        gt_categories = objects.get("ref") or [row["anomaly_name"]]
        if not isinstance(gt_categories, (list, tuple)) or not all(isinstance(c, str) and c for c in gt_categories):
            raise ValueError(f"Invalid reference categories: {key}")
        gt_categories = [aliases.get(c, c) for c in gt_categories]
        gt_boxes = objects.get("bbox") or []
        if not isinstance(gt_boxes, (list, tuple)) or not all(valid_box(b) for b in gt_boxes):
            raise ValueError(f"Invalid reference boxes: {key}")
        first = categories[0] if categories else None
        match = re.match(r"E\d{2}S(\d{2})", row["code"])
        if not match:
            raise ValueError(f"Unknown sample code for {key}")
        measurement = {
            "parsed": categories is not None,
            "first_category_correct": first == gt_name,
            "exact_categories_correct": categories == gt_categories,
            "binary_correct": first is not None and (first == "Normal") == (gt_name == "Normal"),
            "iou": None,
        }
        # Score the first reference box when localization is applicable.
        if gt_boxes and gt_boxes[0] != [0, 0, 0, 0]:
            measurement["iou"] = iou(gt_boxes[0], boxes[0]) if boxes else 0.0
        groups["overall"].append(measurement)
        groups[f"scene_{match.group(1)}"].append(measurement)

    if not seen:
        raise ValueError("No prediction records")

    def summary(measurements):
        count = len(measurements)
        overlaps = [m["iou"] for m in measurements if m["iou"] is not None]
        return {
            "examples": count,
            "parse_failures": sum(not m["parsed"] for m in measurements),
            "first_category_accuracy": sum(m["first_category_correct"] for m in measurements) / count,
            "exact_categories_accuracy": sum(m["exact_categories_correct"] for m in measurements) / count,
            "binary_anomaly_accuracy": sum(m["binary_correct"] for m in measurements) / count,
            "localized_examples": len(overlaps),
            "mean_first_box_iou": sum(overlaps) / len(overlaps) if overlaps else None,
            "localization_hit_rate": {
                str(t): sum(x >= t for x in overlaps) / len(overlaps) if overlaps else None
                for t in [0.25, 0.5, 0.75, 0.9]
            },
        }

    result = {name: summary(values) for name, values in sorted(groups.items())}
    scenes = [values for name, values in result.items() if name != "overall"]
    localized_scenes = [values for values in scenes if values["mean_first_box_iou"] is not None]
    result["overall"].update({
        "scene_count": len(scenes),
        "localized_scene_count": len(localized_scenes),
        "scene_macro_first_category_accuracy": sum(s["first_category_accuracy"] for s in scenes) / len(scenes),
        "scene_macro_mean_first_box_iou": (
            sum(s["mean_first_box_iou"] for s in localized_scenes) / len(localized_scenes)
            if localized_scenes else None
        ),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-aliases", type=Path, help="Optional JSON mapping; no aliases are applied by default")
    parser.add_argument("--references", type=Path, help="Original annotation split; verify full sample coverage and unchanged ground truth")
    args = parser.parse_args()
    inputs = [p for p in [args.predictions, args.references, args.label_aliases] if p is not None]
    if args.output and any(args.output.resolve() == p.resolve() for p in inputs):
        parser.error("Metrics must not overwrite prediction, reference or alias inputs")
    try:
        aliases = json.loads(args.label_aliases.read_text()) if args.label_aliases else None
        references = list(read_jsonl(args.references)) if args.references else None
        result = evaluate(read_jsonl(args.predictions), aliases, references)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.references:
        scope = {"reference_file": args.references.name,
                 "reference_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest(),
                 "reference_examples": len(references), "coverage_verified": True,
                 "scene_ids": sorted(k for k in result if k.startswith("scene_"))}
        result["overall"]["evaluation_scope"] = scope
    if args.label_aliases:
        result["overall"]["label_aliases"] = aliases
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
