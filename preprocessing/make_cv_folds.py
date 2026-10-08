#!/usr/bin/env python3
"""Build patient-level cross-validation folds from one sliced SegTHOR dataset.

The slices of all patients are expected in ``<source>/{img,gt}`` (as produced by
``slice_segthor.py --retains 0``, in its ``train`` folder). Patients are shuffled
with a fixed seed and split into ``--folds`` groups of near-equal size. For each
fold k, ``<dest>/fold_k/{train,val}/{img,gt}`` is created with relative symlinks
to the source slices, so the slices are stored only once and every fold can be
used as a regular dataset folder (``main.py --data-dir <dest>/fold_k``).

The patient lists are also written to ``<dest>/folds.json``.
"""

import argparse
import json
import os
import random
from pathlib import Path


def patient_of(path: Path) -> str:
    # Patient_01_0123.png -> Patient_01
    return path.stem.rsplit("_", 1)[0]


def make_folds(patients: list[str], n_folds: int, seed: int) -> list[list[str]]:
    shuffled = sorted(patients)
    random.Random(seed).shuffle(shuffled)
    # Round-robin, so fold sizes differ by at most one patient
    return [sorted(shuffled[k::n_folds]) for k in range(n_folds)]


def link_slices(source: Path, dest: Path, patients: set[str]) -> int:
    count = 0
    for sub in ["img", "gt"]:
        (dest / sub).mkdir(parents=True)
        for path in sorted((source / sub).glob("*.png")):
            if patient_of(path) in patients:
                (dest / sub / path.name).symlink_to(os.path.relpath(path, dest / sub))
                count += sub == "img"
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder with img/ and gt/ of all patients")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    images = sorted((args.source / "img").glob("*.png"))
    labels = sorted((args.source / "gt").glob("*.png"))
    if not images:
        raise RuntimeError(f"No slices found in {args.source / 'img'}")
    if [p.name for p in images] != [p.name for p in labels]:
        raise RuntimeError("img/ and gt/ do not contain the same slice names")
    if args.dest.exists():
        raise RuntimeError(f"Destination already exists, refusing to overwrite: {args.dest}")

    patients = sorted({patient_of(p) for p in images})
    folds = make_folds(patients, args.folds, args.seed)
    print(f"{len(patients)} patients, {len(images)} slices, {args.folds} folds")

    for k, val_patients in enumerate(folds):
        train_patients = [p for p in patients if p not in val_patients]
        fold_dir = args.dest / f"fold_{k}"
        n_train = link_slices(args.source, fold_dir / "train", set(train_patients))
        n_val = link_slices(args.source, fold_dir / "val", set(val_patients))
        assert n_train + n_val == len(images)
        print(f"fold_{k}: {len(train_patients)} train patients ({n_train} slices), "
              f"{len(val_patients)} val patients ({n_val} slices): {', '.join(val_patients)}")

    with open(args.dest / "folds.json", "w") as f:
        json.dump({"seed": args.seed,
                   "source": str(args.source.resolve()),
                   "folds": [{"val": val, "train": [p for p in patients if p not in val]}
                             for val in folds]},
                  f, indent=2)
    print(f"Saved fold assignment to {args.dest / 'folds.json'}")


if __name__ == "__main__":
    main()
