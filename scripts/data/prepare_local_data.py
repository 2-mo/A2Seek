"""Create private training files containing resolved local image paths."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ROOT

import argparse

from a2seek.data import DEFAULT_VARIANT, get_dataset_root, localize, read_jsonl, write_jsonl
from a2seek.reward_core import training_solution
from a2seek.media import materialize_images


def prepare_row(row, media_root):
    result = localize(row, media_root)
    # Grounding placeholders need explicit objects even for Normal training targets.
    if result["anomaly_name"] == "Normal" and not result.get("objects"):
        result["objects"] = {"ref": ["Normal"], "bbox": [[0, 0, 0, 0]]}
    result["solution"] = training_solution(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="External repository containing dataset/, annotations/ and images/")
    parser.add_argument("--media-root", type=Path, help="Defaults to DATASET_ROOT/images")
    parser.add_argument("--variants", nargs="+", default=[DEFAULT_VARIANT])
    parser.add_argument("--output", type=Path, default=ROOT / ".local/dataset")
    args = parser.parse_args()
    data_root = get_dataset_root(args.dataset_root)
    media_root = args.media_root or data_root / "images"
    if not (data_root / "dataset").is_dir():
        parser.error("Set --dataset-root or A2SEEK_DATA_ROOT to the prepared dataset repository")
    if not args.media_root:
        materialize_images(data_root)
    variants = sorted(p.name for p in (data_root / "dataset").iterdir() if p.is_dir()) if args.variants == ["all"] else args.variants
    for variant in variants:
        folder = data_root / "dataset" / variant
        if not folder.is_dir() or Path(variant).name != variant:
            parser.error(f"Unknown dataset variant: {variant}")
        target = (args.output / variant).resolve()
        if any(target.is_relative_to((data_root / name).resolve()) for name in ("dataset", "annotations")):
            parser.error("Output must not overwrite published annotations")
        for path in sorted(folder.glob("*.jsonl")):
            write_jsonl(target / path.name, (prepare_row(row, media_root) for row in read_jsonl(path)))
        print(f"Prepared {variant}: {target}")


if __name__ == "__main__":
    main()
