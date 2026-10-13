#!/usr/bin/env python3
"""Postprocess held-out PNG slices or NIfTI predictions and compare with NIfTI GT.

PNG predictions are read from ``input_dir/<model>/*.png``; when a model has no
PNG files, its ``*.nii`` and ``*.nii.gz`` predictions are used instead.
Ground truth is matched by patient name and read only from
``gt_dir/<Patient_**>/GT.nii.gz``. Per-model processed NIfTIs and before/after
Dice and HD95 summaries are written under ``output_dir``.
"""

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

if __package__:
    from .connected_compontents import (
        keep_largest_components,
        load_volumes,
    )
    from .nifti_utils import (
        load_nifti_labels,
        save_nifti_labels,
        slices_to_reference_volume,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from postprocessing.connected_compontents import (
        keep_largest_components,
        load_volumes,
    )
    from postprocessing.nifti_utils import (
        load_nifti_labels,
        save_nifti_labels,
        slices_to_reference_volume,
    )

from utils import PatientVolumeDice, class2one_hot, surface_hd95


def patient_dice_by_class(
    prediction: np.ndarray, target: np.ndarray, classes: int
) -> np.ndarray:
    """Compute one patient-volume Dice per class using the shared metric."""
    metric = PatientVolumeDice(classes)
    prediction_labels = torch.from_numpy(prediction.astype(np.int64, copy=False))[None, ...]
    target_labels = torch.from_numpy(target.astype(np.int64, copy=False))[None, ...]
    metric.update(
        class2one_hot(prediction_labels, classes),
        class2one_hot(target_labels, classes),
        ["comparison_patient"],
    )
    _, dice = metric.compute()
    return dice[0].numpy()


def metric_delta(before: float, after: float) -> float:
    if np.isnan(before) or np.isnan(after):
        return float("nan")
    return after - before


def safe_mean(values: list[float]) -> float:
    defined_values = [float(value) for value in values if not np.isnan(value)]
    return float(np.mean(defined_values)) if defined_values else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ground_truth_paths(gt_root: Path) -> dict[str, Path]:
    if not gt_root.is_dir():
        raise FileNotFoundError(f"Ground-truth root does not exist: {gt_root}")
    paths = sorted(path for path in gt_root.rglob("GT.nii.gz") if path.is_file())
    if not paths:
        raise RuntimeError(f"No <Patient_**>/GT.nii.gz files found under {gt_root}")
    matches: dict[str, Path] = {}
    for path in paths:
        patient = path.parent.name.casefold()
        if patient in matches:
            raise ValueError(
                f"Multiple GT.nii.gz files found for {path.parent.name}: "
                f"{matches[patient]} and {path}"
            )
        matches[patient] = path
    return matches


def nifti_patient_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/heldout_predictions"),
        help="Root with one model directory of PNG slices or NIfTI predictions per experiment.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/segthor_train_full/train"),
        help="Root with patient subdirectories containing GT.nii.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/heldout_comparison"),
        help="Destination for postprocessed NIfTIs and comparison CSV files.",
    )
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument(
        "--label-step",
        type=float,
        default=63.0,
        help="PNG label encoding multiplier; 63 encodes labels 0..4 as 0,63,...,252.",
    )
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="HD95 spacing in GT array-axis order; defaults to each GT NIfTI header.",
    )
    args = parser.parse_args()

    if args.classes < 2 or args.classes > 256:
        parser.error("--classes must be between 2 and 256")
    if args.label_step <= 0:
        parser.error("--label-step must be positive")
    if args.spacing is not None and (
        not np.all(np.isfinite(args.spacing)) or np.any(np.asarray(args.spacing) <= 0)
    ):
        parser.error("--spacing values must be positive and finite")
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist or is not a directory: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from --input-dir")
    if args.gt_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from --gt-dir")

    gt_paths = ground_truth_paths(args.gt_dir)
    model_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    model_predictions: dict[
        str, tuple[dict[str, np.ndarray], bool, dict[str, np.ndarray]]
    ] = {}
    for model_dir in model_dirs:
        png_paths = sorted(model_dir.glob("*.png"))
        if png_paths:
            predictions, _ = load_volumes(model_dir, args.classes, args.label_step)
            model_predictions[model_dir.name] = (
                {patient.casefold(): volume for patient, volume in predictions.items()},
                False,
                {},
            )
            continue

        nifti_paths = sorted(
            [*model_dir.glob("*.nii"), *model_dir.glob("*.nii.gz")]
        )
        if not nifti_paths:
            print(f"Skipping {model_dir.name}: no PNG slices or NIfTI predictions")
            continue
        predictions: dict[str, np.ndarray] = {}
        prediction_affines: dict[str, np.ndarray] = {}
        for prediction_path in nifti_paths:
            patient_key = nifti_patient_id(prediction_path).casefold()
            if patient_key in predictions:
                raise ValueError(
                    f"Multiple NIfTI predictions found for "
                    f"{model_dir.name}/{patient_key}"
                )
            prediction, prediction_image = load_nifti_labels(
                prediction_path, args.classes
            )
            predictions[patient_key] = prediction
            prediction_affines[patient_key] = prediction_image.affine
        model_predictions[model_dir.name] = (
            predictions,
            True,
            prediction_affines,
        )
    if not model_predictions:
        raise RuntimeError(f"No model prediction PNGs or NIfTIs found under {args.input_dir}")

    patient_sets = [set(predictions) for predictions, _, _ in model_predictions.values()]
    if any(patient_set != patient_sets[0] for patient_set in patient_sets[1:]):
        raise ValueError("Model prediction directories do not contain the same patient set")

    per_patient_class_rows: list[dict[str, object]] = []
    per_patient_rows: list[dict[str, object]] = []
    processed_root = args.output_dir / "processed_predictions"

    for model_name, (
        predictions,
        nifti_input,
        prediction_affines,
    ) in model_predictions.items():
        missing_gt = sorted(set(predictions) - gt_paths.keys())
        if missing_gt:
            raise ValueError(f"{model_name}: no GT.nii.gz found for patients {missing_gt}")

        for patient_key in sorted(predictions):
            gt_path = gt_paths[patient_key]
            patient_id = gt_path.parent.name
            gt_image = nib.load(str(gt_path))
            target = np.asanyarray(gt_image.dataobj)
            if target.ndim != 3:
                raise ValueError(f"Expected 3D ground truth at {gt_path}, got {target.shape}")
            if not np.all(np.isfinite(target)) or not np.all(target == np.rint(target)):
                raise ValueError(f"Ground-truth labels must be finite integers in {gt_path}")
            if target.size and (target.min() < 0 or target.max() >= args.classes):
                raise ValueError(
                    f"Ground-truth labels in {gt_path} must be between 0 and "
                    f"{args.classes - 1}; found {np.unique(target)}"
                )
            target = target.astype(np.uint8)
            print(f"Matching {model_name}/{patient_id} -> {gt_path}")

            prediction_volume = predictions[patient_key]
            filtered_zyx = keep_largest_components(
                prediction_volume,
                args.classes,
                args.connectivity,
            )
            if nifti_input:
                if prediction_volume.shape != target.shape:
                    raise ValueError(
                        f"Prediction/ground-truth shape mismatch for "
                        f"{model_name}/{patient_id}: {prediction_volume.shape} "
                        f"and {target.shape}"
                    )
                if not np.allclose(
                    prediction_affines[patient_key], gt_image.affine
                ):
                    raise ValueError(
                        f"Prediction/ground-truth affine mismatch for "
                        f"{model_name}/{patient_id}"
                    )
                before = prediction_volume
                after = filtered_zyx
            else:
                before = slices_to_reference_volume(prediction_volume, target.shape)
                after = slices_to_reference_volume(filtered_zyx, target.shape)

            spacing = (
                tuple(float(value) for value in args.spacing)
                if args.spacing is not None
                else tuple(float(value) for value in gt_image.header.get_zooms()[:3])
            )
            dice_before = patient_dice_by_class(before, target, args.classes)
            dice_after = patient_dice_by_class(after, target, args.classes)
            dice_before_values: list[float] = []
            dice_after_values: list[float] = []
            hd95_before_values: list[float] = []
            hd95_after_values: list[float] = []

            for class_index in range(1, args.classes):
                class_dice_before = float(dice_before[class_index])
                class_dice_after = float(dice_after[class_index])
                class_hd95_before = surface_hd95(
                    before == class_index, target == class_index, spacing
                )
                class_hd95_after = surface_hd95(
                    after == class_index, target == class_index, spacing
                )
                per_patient_class_rows.append({
                    "model": model_name,
                    "patient": patient_id,
                    "ground_truth": str(gt_path),
                    "class": class_index,
                    "dice_before": class_dice_before,
                    "dice_after": class_dice_after,
                    "dice_delta_after_minus_before": metric_delta(
                        class_dice_before, class_dice_after
                    ),
                    "hd95_before": class_hd95_before,
                    "hd95_after": class_hd95_after,
                    "hd95_delta_after_minus_before": metric_delta(
                        class_hd95_before, class_hd95_after
                    ),
                })
                if not np.isnan(class_dice_before):
                    dice_before_values.append(class_dice_before)
                if not np.isnan(class_dice_after):
                    dice_after_values.append(class_dice_after)
                if not np.isnan(class_hd95_before):
                    hd95_before_values.append(class_hd95_before)
                if not np.isnan(class_hd95_after):
                    hd95_after_values.append(class_hd95_after)

            mean_dice_before = safe_mean(dice_before_values)
            mean_dice_after = safe_mean(dice_after_values)
            mean_hd95_before = safe_mean(hd95_before_values)
            mean_hd95_after = safe_mean(hd95_after_values)
            per_patient_rows.append({
                "model": model_name,
                "patient": patient_id,
                "ground_truth": str(gt_path),
                "mean_foreground_dice_before": mean_dice_before,
                "mean_foreground_dice_after": mean_dice_after,
                "mean_dice_delta_after_minus_before": metric_delta(
                    mean_dice_before, mean_dice_after
                ),
                "mean_foreground_hd95_before": mean_hd95_before,
                "mean_foreground_hd95_after": mean_hd95_after,
                "mean_hd95_delta_after_minus_before": metric_delta(
                    mean_hd95_before, mean_hd95_after
                ),
            })

            reference = gt_image
            output_path = processed_root / model_name / f"{patient_id}.nii.gz"
            save_nifti_labels(after, reference, output_path)
            print(f"Saved: {output_path}")

        print(f"{model_name}: processed and evaluated {len(predictions)} patients")

    model_rows: list[dict[str, object]] = []
    model_class_rows: list[dict[str, object]] = []
    model_names = sorted({str(row["model"]) for row in per_patient_rows})
    for model_name in model_names:
        patient_rows = [row for row in per_patient_rows if row["model"] == model_name]
        for class_index in range(1, args.classes):
            class_rows = [
                row for row in per_patient_class_rows
                if row["model"] == model_name and row["class"] == class_index
            ]
            class_dice_before = safe_mean([float(row["dice_before"]) for row in class_rows])
            class_dice_after = safe_mean([float(row["dice_after"]) for row in class_rows])
            class_hd95_before = safe_mean([float(row["hd95_before"]) for row in class_rows])
            class_hd95_after = safe_mean([float(row["hd95_after"]) for row in class_rows])
            model_class_rows.append({
                "model": model_name,
                "class": class_index,
                "n_patients_with_class": sum(
                    not np.isnan(float(row["dice_before"])) for row in class_rows
                ),
                "mean_patient_dice_before": class_dice_before,
                "mean_patient_dice_after": class_dice_after,
                "mean_patient_dice_delta": metric_delta(class_dice_before, class_dice_after),
                "mean_patient_hd95_before": class_hd95_before,
                "mean_patient_hd95_after": class_hd95_after,
                "mean_patient_hd95_delta": metric_delta(class_hd95_before, class_hd95_after),
            })

        model_dice_before = safe_mean([
            float(row["mean_foreground_dice_before"]) for row in patient_rows
        ])
        model_dice_after = safe_mean([
            float(row["mean_foreground_dice_after"]) for row in patient_rows
        ])
        model_hd95_before = safe_mean([
            float(row["mean_foreground_hd95_before"]) for row in patient_rows
        ])
        model_hd95_after = safe_mean([
            float(row["mean_foreground_hd95_after"]) for row in patient_rows
        ])
        model_rows.append({
            "model": model_name,
            "n_patients": len(patient_rows),
            "mean_patient_foreground_dice_before": model_dice_before,
            "mean_patient_foreground_dice_after": model_dice_after,
            "mean_patient_dice_delta": metric_delta(model_dice_before, model_dice_after),
            "mean_patient_foreground_hd95_before": model_hd95_before,
            "mean_patient_foreground_hd95_after": model_hd95_after,
            "mean_patient_foreground_hd95_delta": metric_delta(
                model_hd95_before, model_hd95_after
            ),
        })

    overall_class_rows: list[dict[str, object]] = []
    for class_index in range(1, args.classes):
        class_rows = [
            row for row in per_patient_class_rows if row["class"] == class_index
        ]
        class_dice_before = safe_mean([float(row["dice_before"]) for row in class_rows])
        class_dice_after = safe_mean([float(row["dice_after"]) for row in class_rows])
        class_hd95_before = safe_mean([float(row["hd95_before"]) for row in class_rows])
        class_hd95_after = safe_mean([float(row["hd95_after"]) for row in class_rows])
        overall_class_rows.append({
            "class": class_index,
            "n_model_patient_cases": len(class_rows),
            "mean_patient_dice_before": class_dice_before,
            "mean_patient_dice_after": class_dice_after,
            "mean_patient_dice_delta": metric_delta(class_dice_before, class_dice_after),
            "mean_patient_hd95_before": class_hd95_before,
            "mean_patient_hd95_after": class_hd95_after,
            "mean_patient_hd95_delta": metric_delta(class_hd95_before, class_hd95_after),
        })

    write_csv(args.output_dir / "per_patient_per_class.csv", per_patient_class_rows, [
        "model", "patient", "ground_truth", "class", "dice_before", "dice_after",
        "dice_delta_after_minus_before", "hd95_before", "hd95_after",
        "hd95_delta_after_minus_before",
    ])
    write_csv(args.output_dir / "per_patient_summary.csv", per_patient_rows, [
        "model", "patient", "ground_truth", "mean_foreground_dice_before",
        "mean_foreground_dice_after", "mean_dice_delta_after_minus_before",
        "mean_foreground_hd95_before", "mean_foreground_hd95_after",
        "mean_hd95_delta_after_minus_before",
    ])
    write_csv(args.output_dir / "per_model_per_class_summary.csv", model_class_rows, [
        "model", "class", "n_patients_with_class", "mean_patient_dice_before",
        "mean_patient_dice_after", "mean_patient_dice_delta",
        "mean_patient_hd95_before", "mean_patient_hd95_after", "mean_patient_hd95_delta",
    ])
    write_csv(args.output_dir / "per_model_summary.csv", model_rows, [
        "model", "n_patients", "mean_patient_foreground_dice_before",
        "mean_patient_foreground_dice_after", "mean_patient_dice_delta",
        "mean_patient_foreground_hd95_before", "mean_patient_foreground_hd95_after",
        "mean_patient_foreground_hd95_delta",
    ])
    print(f"Saved per-model metric summaries and processed NIfTIs under {args.output_dir}")


if __name__ == "__main__":
    main()
