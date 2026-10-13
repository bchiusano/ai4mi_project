#!/usr/bin/env python3
"""Final postprocessing pipeline for held-out predictions (PNG slices or NIfTI, per model).

Order per patient: largest component (chosen classes) -> bridge gaps -> closing.
PNG models are processed on their slice grid and written as PNG slices; NIfTI models are
processed in their own space and written as NIfTI with the input geometry.

Usage:
    python -m postprocessing.postprocess_pipeline \
        --input-dir results/heldout_predictions --output-dir results/heldout_final \
        --models ce_dice_25d swin_unetr --lcc-classes 2,3 --bridge-classes 1 --close-classes 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .connected_compontents import load_volumes, save_volumes
    from .nifti_utils import load_nifti_labels, save_nifti_labels
    from .steps import process_volume
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from postprocessing.connected_compontents import load_volumes, save_volumes
    from postprocessing.nifti_utils import load_nifti_labels, save_nifti_labels
    from postprocessing.steps import process_volume


def parse_classes(text: str, classes: int, parser: argparse.ArgumentParser, name: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    for value in values:
        if not 1 <= value < classes:
            parser.error(f"{name}: class {value} must be between 1 and {classes - 1}")
    return values


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def report(model: str, patient: str, log: dict[str, int]) -> None:
    changes = ", ".join(f"{step}: {count}" for step, count in log.items()) or "no steps"
    print(f"{model}/{patient}  voxels changed -> {changes}", flush=True)


def run_png_model(model_dir: Path, output_dir: Path, args, params: dict) -> None:
    volumes, paths_by_patient = load_volumes(model_dir, args.classes, args.label_step)
    processed = {}
    for patient, volume in sorted(volumes.items()):
        processed[patient], log = process_volume(volume, **params)  # volume is (z, rows, cols)
        report(model_dir.name, patient, log)
    save_volumes(processed, paths_by_patient, output_dir, args.label_step)


def run_nifti_model(model_dir: Path, output_dir: Path, args, params: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for path in sorted([*model_dir.glob("*.nii"), *model_dir.glob("*.nii.gz")]):
        stem = nifti_stem(path)
        if stem in seen:
            raise ValueError(f"Multiple NIfTI predictions for {model_dir.name}/{stem}")
        seen.add(stem)
        labels_xyz, image = load_nifti_labels(path, args.classes)
        volume_zxy = np.transpose(labels_xyz, (2, 0, 1))  # steps expect the slice axis first
        processed, log = process_volume(volume_zxy, **params)
        save_nifti_labels(np.transpose(processed, (1, 2, 0)), image, output_dir / f"{stem}.nii.gz")
        report(model_dir.name, stem, log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", type=Path, required=True, help="Root with one folder per model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Model folder names to process.")
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--label-step", type=float, default=63.0)
    parser.add_argument("--lcc-classes", default="2,3", help="Comma-separated classes for largest component.")
    parser.add_argument("--bridge-classes", default="1", help="Comma-separated classes to bridge gaps in.")
    parser.add_argument("--bridge-min-size", type=int, default=200, help="Ignore pieces below this many voxels.")
    parser.add_argument("--bridge-max-gap", type=int, default=30, help="Longest gap (slices) to bridge.")
    parser.add_argument("--close-classes", default="1", help="Comma-separated classes to close along z.")
    parser.add_argument("--close-z", type=int, default=5)
    parser.add_argument("--close-xy", type=int, default=3)
    args = parser.parse_args()

    if not 2 <= args.classes <= 256:
        parser.error("--classes must be between 2 and 256")
    if args.label_step <= 0:
        parser.error("--label-step must be positive")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from --input-dir")
    params = {
        "lcc_classes": parse_classes(args.lcc_classes, args.classes, parser, "--lcc-classes"),
        "bridge_classes": parse_classes(args.bridge_classes, args.classes, parser, "--bridge-classes"),
        "bridge_min_size": args.bridge_min_size,
        "bridge_max_gap": args.bridge_max_gap,
        "close_classes": parse_classes(args.close_classes, args.classes, parser, "--close-classes"),
        "close_z": args.close_z,
        "close_xy": args.close_xy,
    }

    for model in args.models:
        model_dir = args.input_dir / model
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model folder not found: {model_dir}")
        has_png = any(model_dir.glob("*.png"))
        has_nifti = any(model_dir.glob("*.nii")) or any(model_dir.glob("*.nii.gz"))
        if has_png:
            run_png_model(model_dir, args.output_dir / model, args, params)
        elif has_nifti:
            run_nifti_model(model_dir, args.output_dir / model, args, params)
        else:
            raise RuntimeError(f"No PNG or NIfTI predictions in {model_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "pipeline_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved postprocessed predictions and pipeline_config.json under {args.output_dir}")


if __name__ == "__main__":
    main()