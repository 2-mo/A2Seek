"""Restore JPEGs from the dataset's Parquet image bank."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ROOT

import argparse

from a2seek.data import get_dataset_root
from a2seek.media import materialize_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    count = materialize_images(get_dataset_root(args.dataset_root))
    print(f"Images ready: {count:,}")


if __name__ == "__main__":
    main()
