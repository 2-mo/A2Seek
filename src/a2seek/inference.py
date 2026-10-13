# Copyright (c) Alibaba, Inc. and its affiliates.
"""Run Qwen2.5-VL inference on a published annotation split."""

import argparse
import hashlib
import json
from itertools import islice
import os
from pathlib import Path

from a2seek.paths import CONFIG_ROOT, DEFAULT_MODEL_ROOT, PROJECT_ROOT as ROOT
from a2seek.data import DEFAULT_VARIANT, get_dataset_root, inference_messages, localize, read_jsonl, sample_id, validate_single_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    preset_root = CONFIG_ROOT / "inference"
    parser.add_argument("--preset", choices=sorted(p.stem for p in preset_root.glob("*.json")), help="Optional zero-shot inference configuration")
    parser.add_argument("--model",
                        help="Merged checkpoint directory or Hugging Face model ID (default: A2SEEK_MODEL_ROOT or models/A2Seek-R1-3B)")
    parser.add_argument("--adapter", help="Optional LoRA checkpoint")
    parser.add_argument("--dataset-root", type=Path, help="Dataset snapshot root (default: A2SEEK_DATA_ROOT, .local/dataset-root or data/A2Seek)")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/predictions.jsonl")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpus", help="Set CUDA_VISIBLE_DEVICES for this process")
    parser.add_argument("--resume", action="store_true", help="Continue a matching prediction file")
    parser.add_argument("--num-shards", type=int, default=1, help="Partition input rows across independent inference processes")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based index of this inference partition")
    args = parser.parse_args()
    preset = json.loads((preset_root / (args.preset + ".json")).read_text()) if args.preset else {}
    args.model = args.model or preset.get("model") or os.environ.get("A2SEEK_MODEL_ROOT") or str(DEFAULT_MODEL_ROOT)
    args.max_tokens = args.max_tokens if args.max_tokens is not None else preset.get("max_tokens", 1024)
    args.batch_size = args.batch_size if args.batch_size is not None else preset.get("batch_size", 1)
    if preset.get("type") == "zero_shot" and args.adapter:
        parser.error("Zero-shot presets use an unadapted base model; omit --adapter")
    data_root = get_dataset_root(args.dataset_root)
    args.data = args.data or data_root / "dataset" / preset.get("variant", DEFAULT_VARIANT) / (preset.get("split", "validation") + ".jsonl")
    args.media_root = (args.media_root or data_root / "images").resolve()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1 or args.max_tokens < 1:
        parser.error("--batch-size and --max-tokens must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.output.resolve() == args.data.resolve():
        parser.error("Predictions must not overwrite input annotations")
    if args.media_root == (data_root / "images").resolve():
        from a2seek.media import materialize_images
        materialize_images(data_root)
    # Check media before allocating a model or touching prediction outputs.
    rows = [localize(row, args.media_root) for i, row in enumerate(islice(read_jsonl(args.data), args.limit))
            if i % args.num_shards == args.shard_index]
    try:
        rows = [validate_single_frame(row) for row in rows]
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if preset:
        prompt = (CONFIG_ROOT.parent / preset["prompt_file"]).read_text().rstrip("\n")
        if any(inference_messages(row) != [{"role": "user", "content": prompt}] for row in rows):
            parser.error("Input prompts do not match the selected zero-shot preset; use its dataset view")
    completed = 0
    def model_ref(value):
        if value is None:
            return None
        path = Path(value)
        if path.is_dir():
            return {"name": path.name, "source_id": hashlib.sha256(str(path.resolve()).encode()).hexdigest()}
        return {"name": value}

    settings = {"model": model_ref(args.model), "adapter": model_ref(args.adapter),
                "max_tokens": args.max_tokens, "temperature": 0, "batch_size": args.batch_size,
                "num_shards": args.num_shards, "shard_index": args.shard_index}
    if args.adapter:
        settings["adapter_loader"] = "qwen-lora-v1"
    if preset:
        settings["preset"] = args.preset
        settings["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    if args.resume and args.output.exists():
        previous = list(read_jsonl(args.output))
        if len(previous) > len(rows) or any(sample_id(saved) != sample_id(row) or saved.get("inference") != settings
                                            for saved, row in zip(previous, rows)):
            parser.error("Existing predictions do not match this model, settings and sample order")
        if any(inference_messages(saved) != inference_messages(row)
               or localize(saved, args.media_root)["images"] != row["images"]
               for saved, row in zip(previous, rows)):
            parser.error("Existing predictions do not match the input prompts and frame sequence")
        completed = len(previous)
    if completed == len(rows):
        print(f"Already wrote {completed} predictions to {args.output}")
        return
    os.environ.setdefault("USE_HF", "1")
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    engine = None
    def generate(requests):
        nonlocal engine
        from swift.llm import InferRequest, PtEngine, RequestConfig, get_model_tokenizer, get_template
        if engine is None:
            from a2seek.adapters import load_lora
            model, tokenizer = get_model_tokenizer(args.model, model_type="qwen2_5_vl")
            if args.adapter:
                model = load_lora(model, args.adapter)
            template = get_template("qwen2_5_vl", tokenizer, default_system="")
            engine = PtEngine.from_model_template(model, template, max_batch_size=args.batch_size)
        responses = engine.infer([InferRequest(**r) for r in requests], RequestConfig(max_tokens=args.max_tokens, temperature=0))
        if len(responses) != len(requests):
            raise RuntimeError("Inference returned an unexpected number of responses")
        return [r.choices[0].message.content for r in responses]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as stream:
        offset = completed
        while offset < len(rows):
            end = min(offset + args.batch_size, len(rows))
            batch = rows[offset:end]
            predictions = generate([{"messages": inference_messages(row), "images": row["images"]} for row in batch])
            records = [dict(row, prediction=prediction, inference=settings) for row, prediction in zip(batch, predictions)]
            for row, record in zip(batch, records):
                # Prediction artifacts retain portable media paths and model references.
                record["images"] = ["images/" + Path(p).relative_to(args.media_root).as_posix() for p in row["images"]]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            offset = end
            print(f"Predictions: {offset}/{len(rows)}", flush=True)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
