#!/usr/bin/env python3

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from utils import patient_id_from_stem, surface_hd95


ORGAN_NAMES = ("background", "esophagus", "heart", "trachea", "aorta")


def patient_ids_from_split(split_path: Path, subset: str) -> set[str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return set(split[subset])


def matched_prediction_and_ground_truth(
    pred_dir: Path,
    gt_dir: Path,
    expected_patient_ids: set[str] | None,
) -> tuple[dict[str, Path], dict[str, Path]]:
    predictions = {path.name: path for path in pred_dir.glob("*.png")}
    ground_truth = {path.name: path for path in gt_dir.glob("*.png")}
    if not predictions:
        raise RuntimeError(f"No prediction PNG files in {pred_dir}")
    if expected_patient_ids is not None:
        ground_truth = {
            name: path
            for name, path in ground_truth.items()
            if patient_id_from_stem(path.stem) in expected_patient_ids
        }
    if predictions.keys() != ground_truth.keys():
        missing_predictions = sorted(ground_truth.keys() - predictions.keys())
        missing_ground_truth = sorted(predictions.keys() - ground_truth.keys())
        raise RuntimeError(
            f"Prediction/GT names differ. Missing predictions={missing_predictions[:5]}, "
            f"missing ground truth={missing_ground_truth[:5]}"
        )
    return predictions, ground_truth


def decode_mask(path: Path, classes: int, label_step: float) -> np.ndarray:
    image = np.asarray(Image.open(path))
    labels = np.rint(image.astype(np.float64) / label_step).astype(np.int64)
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError(f"Unexpected labels in {path}: {np.unique(image)}")
    return labels


def evaluate_prediction_directory(
    pred_dir: Path,
    gt_dir: Path,
    classes: int = 5,
    label_step: float = 63.0,
    expected_patient_ids: set[str] | None = None,
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    """Return patient-level 3D Dice and detailed rows for one prediction folder."""
    predictions, ground_truth = matched_prediction_and_ground_truth(
        pred_dir, gt_dir, expected_patient_ids
    )

    intersections: dict[str, np.ndarray] = {}
    prediction_sizes: dict[str, np.ndarray] = {}
    target_sizes: dict[str, np.ndarray] = {}

    for name in sorted(predictions):
        pred = decode_mask(predictions[name], classes, label_step)
        target = decode_mask(ground_truth[name], classes, label_step)
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch for {name}: {pred.shape} != {target.shape}")

        patient = patient_id_from_stem(Path(name).stem)
        intersections.setdefault(patient, np.zeros(classes, dtype=np.int64))
        prediction_sizes.setdefault(patient, np.zeros(classes, dtype=np.int64))
        target_sizes.setdefault(patient, np.zeros(classes, dtype=np.int64))

        # Rows are target classes and columns are predicted classes.  A single
        # bincount supplies all intersections and class volumes without making
        # one full-size boolean array per class.
        confusion = np.bincount(
            classes * target.ravel() + pred.ravel(),
            minlength=classes * classes,
        ).reshape(classes, classes)
        intersections[patient] += np.diag(confusion)
        prediction_sizes[patient] += confusion.sum(axis=0)
        target_sizes[patient] += confusion.sum(axis=1)

    patient_ids = sorted(intersections)
    dice = np.full((len(patient_ids), classes), np.nan, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for patient_index, patient in enumerate(patient_ids):
        cardinality = prediction_sizes[patient] + target_sizes[patient]
        present = cardinality > 0
        dice[patient_index, present] = 2 * intersections[patient][present] / cardinality[present]
        for class_index in range(classes):
            organ = ORGAN_NAMES[class_index] if class_index < len(ORGAN_NAMES) else f"class_{class_index}"
            rows.append({
                "patient": patient,
                "class": class_index,
                "organ": organ,
                "dice": dice[patient_index, class_index],
                "prediction_voxels": prediction_sizes[patient][class_index],
                "ground_truth_voxels": target_sizes[patient][class_index],
            })

    return patient_ids, dice, rows


def evaluate_hd95_directory(
    pred_dir: Path,
    gt_dir: Path,
    spacing_by_patient: Mapping[str, Sequence[float]],
    classes: int = 5,
    label_step: float = 63.0,
    in_plane_spacing_scale: float = 1.0,
    expected_patient_ids: set[str] | None = None,
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    """Return patient-level 3D surface HD95 values in millimetres.

    Slices are stacked on the third axis to reconstruct ``(x, y, z)`` patient
    volumes.  ``in_plane_spacing_scale`` supports legacy ``spacing.pkl`` files
    created before resized PNG spacing was recorded correctly; for the usual
    512-to-256 preprocessing those files require a scale of 2.
    """
    if not np.isfinite(in_plane_spacing_scale) or in_plane_spacing_scale <= 0:
        raise ValueError("in_plane_spacing_scale must be positive and finite")

    predictions, ground_truth = matched_prediction_and_ground_truth(
        pred_dir, gt_dir, expected_patient_ids
    )

    names_by_patient: dict[str, list[str]] = defaultdict(list)
    for name in predictions:
        names_by_patient[patient_id_from_stem(Path(name).stem)].append(name)

    patient_ids = sorted(names_by_patient)
    hd95 = np.full((len(patient_ids), classes), np.nan, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for patient_index, patient in enumerate(patient_ids):
        if patient not in spacing_by_patient:
            raise KeyError(f"No voxel spacing found for {patient}")
        spacing = np.asarray(spacing_by_patient[patient], dtype=np.float64)
        if spacing.shape != (3,):
            raise ValueError(f"Expected three spacing values for {patient}: {spacing}")
        spacing = spacing.copy()
        spacing[:2] *= in_plane_spacing_scale

        names = sorted(names_by_patient[patient])
        pred_volume = np.stack(
            [decode_mask(predictions[name], classes, label_step) for name in names],
            axis=2,
        )
        target_volume = np.stack(
            [decode_mask(ground_truth[name], classes, label_step) for name in names],
            axis=2,
        )
        for class_index in range(classes):
            # Background HD95 is not an organ-boundary metric and adds a large
            # unnecessary distance transform, so keep it undefined.
            value = (
                float("nan")
                if class_index == 0
                else surface_hd95(
                    pred_volume == class_index,
                    target_volume == class_index,
                    spacing,
                )
            )
            hd95[patient_index, class_index] = value
            organ = (
                ORGAN_NAMES[class_index]
                if class_index < len(ORGAN_NAMES)
                else f"class_{class_index}"
            )
            rows.append({
                "patient": patient,
                "class": class_index,
                "organ": organ,
                "hd95_mm": value,
            })

    return patient_ids, hd95, rows


def merge_metric_rows(
    dice_rows: list[dict[str, object]],
    hd95_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    hd95_lookup = {
        (row["patient"], row["class"]): row["hd95_mm"] for row in hd95_rows
    }
    merged: list[dict[str, object]] = []
    for row in dice_rows:
        combined = dict(row)
        combined["hd95_mm"] = hd95_lookup[(row["patient"], row["class"])]
        merged.append(combined)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved 2D predictions as patient-level 3D SegTHOR volumes"
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--label-step", type=float, default=63.0)
    parser.add_argument(
        "--spacing-file",
        type=Path,
        help="Preprocessed dataset spacing.pkl; enables patient-level 3D HD95.",
    )
    parser.add_argument(
        "--in-plane-spacing-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply x/y spacing by this value. Use 2 for legacy SegTHOR "
            "spacing.pkl files produced by resizing 512x512 slices to 256x256."
        ),
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--split-file",
        type=Path,
        help="split.json; required when --gt-dir is the pooled label directory.",
    )
    parser.add_argument("--subset", choices=["train", "val", "test"], default="val")
    args = parser.parse_args()

    expected_patient_ids = (
        patient_ids_from_split(args.split_file, args.subset)
        if args.split_file is not None
        else None
    )

    patient_ids, dice, rows = evaluate_prediction_directory(
        args.pred_dir,
        args.gt_dir,
        args.classes,
        args.label_step,
        expected_patient_ids,
    )

    print(f"Prediction directory: {args.pred_dir}")
    print(f"Patients: {', '.join(patient_ids)}")
    print(f"Mean foreground patient-level 3D Dice: {np.nanmean(dice[:, 1:]):.6f}")
    for class_index in range(1, args.classes):
        organ = ORGAN_NAMES[class_index] if class_index < len(ORGAN_NAMES) else f"class_{class_index}"
        print(f"{organ.capitalize()} 3D Dice: {np.nanmean(dice[:, class_index]):.6f}")

    if args.spacing_file:
        with args.spacing_file.open("rb") as spacing_input:
            spacing_by_patient = pickle.load(spacing_input)
        hd95_patients, hd95, hd95_rows = evaluate_hd95_directory(
            args.pred_dir,
            args.gt_dir,
            spacing_by_patient,
            args.classes,
            args.label_step,
            args.in_plane_spacing_scale,
            expected_patient_ids,
        )
        if hd95_patients != patient_ids:
            raise RuntimeError(f"HD95 patients differ: {hd95_patients} != {patient_ids}")
        rows = merge_metric_rows(rows, hd95_rows)
        print(f"Mean foreground patient-level 3D HD95: {np.nanmean(hd95[:, 1:]):.6f} mm")
        for class_index in range(1, args.classes):
            organ = (ORGAN_NAMES[class_index]
                     if class_index < len(ORGAN_NAMES)
                     else f"class_{class_index}")
            print(
                f"{organ.capitalize()} 3D HD95: "
                f"{np.nanmean(hd95[:, class_index]):.6f} mm"
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-patient results to: {args.csv}")


if __name__ == "__main__":
    main()
