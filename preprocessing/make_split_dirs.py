#!/usr/bin/env python3
"""Build train/val/test folders from a sliced dataset and a fixed patient split.

The slices of all patients are expected in ``<source>/{img,gt}``, and the split in
a JSON file with one list of patient ids per subset (``{"train": [...], "val": [...],
"test": [...]}``, as in ``SEGTHOR_FULL_PREPROCESSED/split.json``). For each subset,
``<dest>/<subset>/{img,gt}`` is created with relative symlinks to the source slices,
so the folder can be used as a regular dataset folder (``main.py --data-dir <dest>``).

Run from the repository root: python -m preprocessing.make_split_dirs ...
"""

import argparse
import json
from pathlib import Path

from preprocessing.make_cv_folds import link_slices, patient_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder with img/ and gt/ of all patients")
    parser.add_argument("--split", type=Path, required=True, help="JSON file with the patient ids per subset")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    images = sorted((args.source / "img").glob("*.png"))
    labels = sorted((args.source / "gt").glob("*.png"))
    if not images:
        raise RuntimeError(f"No slices found in {args.source / 'img'}")
    if [p.name for p in images] != [p.name for p in labels]:
        raise RuntimeError("img/ and gt/ do not contain the same slice names")
    if args.dest.exists():
        raise RuntimeError(f"Destination already exists, refusing to overwrite: {args.dest}")

    split = json.loads(args.split.read_text())
    subsets: dict[str, list[str]] = {name: split[name] for name in args.subsets}

    all_split = [p for patients in subsets.values() for p in patients]
    if len(all_split) != len(set(all_split)):
        raise RuntimeError("A patient appears in more than one subset")
    on_disk = {patient_of(p) for p in images}
    if missing := set(all_split) - on_disk:
        raise RuntimeError(f"Patients in the split without slices: {sorted(missing)}")
    if unused := on_disk - set(all_split):
        print(f"Warning: patients with slices but in no subset: {sorted(unused)}")

    for name, patients in subsets.items():
        n = link_slices(args.source, args.dest / name, set(patients))
        print(f"{name}: {len(patients)} patients ({n} slices): {', '.join(sorted(patients))}")


if __name__ == "__main__":
    main()
