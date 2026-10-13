"""Run a documented SFT or GRPO preset; use --dry-run to inspect the command."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from a2seek.paths import CONFIG_ROOT, PROJECT_ROOT as ROOT
from a2seek.data import read_jsonl, validate_single_frame
from a2seek.reward_core import ground_truth


def build_command(config, dataset_root, model, output):
    if not model:
        raise ValueError("GRPO requires --model pointing to the merged SFT checkpoint")
    dataset = Path(dataset_root) / config["variant"] / "train.jsonl"
    command = ["swift", config["stage"], "--model", model, "--dataset", str(dataset), "--output_dir", str(output)]
    if config["stage"] == "rlhf":
        command += ["--external_plugins", str(Path(__file__).with_name("rewards.py"))]
    for key, value in config["arguments"].items():
        command.append("--" + key)
        command.extend([str(v) for v in value] if isinstance(value, list) else [str(value)])
    return command, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="sft_3b", choices=sorted(p.stem for p in CONFIG_ROOT.glob("*.json")))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / ".local/dataset",
                        help="Prepared training data directory from scripts/data/prepare_local_data.py")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", help="Set CUDA_VISIBLE_DEVICES for this process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Override the number of optimizer steps, e.g. 1 for a smoke test")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-completion-length", type=int, help="Override GRPO generation length")
    args = parser.parse_args()
    config = json.loads((CONFIG_ROOT / (args.preset + ".json")).read_text())
    for key in ["max_steps", "gradient_accumulation_steps", "max_completion_length"]:
        value = getattr(args, key)
        if value is not None:
            if value < 1:
                parser.error(f"--{key.replace('_', '-')} must be positive")
            if key == "max_completion_length" and config["stage"] != "rlhf":
                parser.error("--max-completion-length is a GRPO option")
            config["arguments"][key] = str(value)
    try:
        command, dataset = build_command(config, args.dataset_root, args.model or config["model"], args.output or ROOT / "outputs" / args.preset)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command))
    if args.dry_run:
        return
    if not dataset.is_file():
        parser.error(f"Missing {dataset}; run scripts/data/prepare_local_data.py first")
    rows = list(read_jsonl(dataset))
    if not rows:
        parser.error("Training data is empty")
    if config.get("num_frames") == 1:
        try:
            for row in rows:
                validate_single_frame(row)
        except (ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(f"Training inputs: {len(rows)} samples, one image per sample")
    if config["stage"] == "rlhf":
        if any(ground_truth(row.get("solution"))[2] is None for row in rows):
            parser.error("GRPO requires structured class/box/seeking targets; run scripts/data/prepare_local_data.py first")
    env = dict(os.environ)
    env.setdefault("USE_HF", "1")
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
