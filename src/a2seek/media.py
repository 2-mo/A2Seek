"""Restore native image paths from a Hugging Face Parquet image bank."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def image_destination(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or len(name.parts) < 2 or name.parts[0] != "images" or ".." in name.parts or "\\" in relative or name.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"Invalid image-bank path: {relative}")
    root = Path(root).resolve()
    target = root.joinpath(*name.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Image-bank path escapes the dataset: {relative}")
    return target


def materialize_images(root):
    root = Path(root).resolve()
    shards = sorted((root / "data/images").glob("*.parquet"))
    if not shards:
        return 0
    import pyarrow.parquet as pq

    stamp = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in shards]
    marker = root / ".local/images-materialized.json"
    if marker.exists() and (root / "images").is_dir():
        previous = json.loads(marker.read_text())
        if previous.get("shards") == stamp:
            return previous["images"]
    count = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                target = image_destination(root, row["path"])
                content = row["image"]["bytes"]
                if content is None or hashlib.sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"Invalid image-bank checksum: {row['path']}")
                if target.exists():
                    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                        raise ValueError(f"Existing image differs: {row['path']}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".part")
                    temporary.write_bytes(content)
                    temporary.replace(target)
                count += 1
        print(f"Restored {count:,} images", flush=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"shards": stamp, "images": count}, indent=2) + "\n")
    return count
