"""Read annotations and resolve media without modifying published records."""

import gzip
import json
import os
from pathlib import Path

from a2seek.paths import DEFAULT_DATASET_ROOT, PROJECT_ROOT


DEFAULT_VARIANT = "cot_single_reflection"


def validate_single_frame(row):
    """Require the image and prompt shape used by the released inference entry point."""
    images = row.get("images", [])
    if len(images) != 1:
        raise ValueError(f"Expected one image per sample for {sample_id(row)}, got {len(images)}")
    messages = inference_messages(row)
    if sum(m["content"].count("<image>") for m in messages) != 1:
        raise ValueError(f"Expected one image tag in the input prompt for {sample_id(row)}")
    return row


def get_dataset_root(value=None):
    """Resolve overrides before falling back to the checkout's data/A2Seek directory."""
    setting = PROJECT_ROOT / ".local/dataset-root"
    selected = value or os.environ.get("A2SEEK_DATA_ROOT")
    if not selected and setting.is_file():
        selected = setting.read_text().strip()
    return Path(selected).expanduser().resolve() if selected else DEFAULT_DATASET_ROOT


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{number}") from exc


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sample_id(row):
    return f"{row['g_id']}/{row['v_id']}/{int(row['f_id']):05d}"


def resolve_image(path, media_root):
    """media_root is the directory containing video_group_*_@_video_* folders."""
    path = Path(path)
    if path.is_absolute():
        return path
    if not path.parts or path.parts[0] != "images" or ".." in path.parts:
        raise ValueError(f"Expected a relative images/... path: {path}")
    return Path(media_root).resolve().joinpath(*path.parts[1:])


def localize(row, media_root, require_images=True):
    result = dict(row)
    paths = [resolve_image(p, media_root) for p in row.get("images", [])]
    if require_images:
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing images for {sample_id(row)}: {missing[:3]}")
    result["images"] = [str(p) for p in paths]
    return result


def inference_messages(row):
    """Drop trailing reference answers before passing a request to the model."""
    messages = [dict(m) for m in row["messages"]]
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"No user prompt for {sample_id(row)}")
    return messages
