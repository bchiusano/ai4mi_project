#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from utils import patient_id_from_stem


ORGAN_NAMES = ("background", "esophagus", "heart", "trachea", "aorta")


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
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    """Return patient-level 3D Dice and detailed rows for one prediction folder."""
    predictions = {p.name: p for p in pred_dir.glob("*.png")}
    ground_truth = {p.name: p for p in gt_dir.glob("*.png")}
    if not predictions:
        raise RuntimeError(f"No prediction PNG files in {pred_dir}")
    if predictions.keys() != ground_truth.keys():
        missing_predictions = sorted(ground_truth.keys() - predictions.keys())
        missing_ground_truth = sorted(predictions.keys() - ground_truth.keys())
        raise RuntimeError(
            f"Prediction/GT names differ. Missing predictions={missing_predictions[:5]}, "
            f"missing ground truth={missing_ground_truth[:5]}"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved 2D predictions as patient-level 3D SegTHOR volumes"
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--label-step", type=float, default=63.0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    patient_ids, dice, rows = evaluate_prediction_directory(
        args.pred_dir, args.gt_dir, args.classes, args.label_step
    )

    print(f"Prediction directory: {args.pred_dir}")
    print(f"Patients: {', '.join(patient_ids)}")
    print(f"Mean foreground patient-level 3D Dice: {np.nanmean(dice[:, 1:]):.6f}")
    for class_index in range(1, args.classes):
        organ = ORGAN_NAMES[class_index] if class_index < len(ORGAN_NAMES) else f"class_{class_index}"
        print(f"{organ.capitalize()} 3D Dice: {np.nanmean(dice[:, class_index]):.6f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-patient results to: {args.csv}")


if __name__ == "__main__":
    main()
