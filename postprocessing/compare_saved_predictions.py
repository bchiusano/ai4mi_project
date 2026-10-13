#!/usr/bin/env python3
"""Compare held-out predictions with saved postprocessed volumes and ground truth."""

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

if __package__:
    from .connected_compontents import load_volumes
    from .nifti_utils import load_nifti_labels, slices_to_reference_volume
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from postprocessing.connected_compontents import load_volumes
    from postprocessing.nifti_utils import load_nifti_labels, slices_to_reference_volume

from utils import PatientVolumeDice, class2one_hot, surface_hd95


def patient_dice_by_class(
    prediction: np.ndarray, target: np.ndarray, classes: int
) -> np.ndarray:
    metric = PatientVolumeDice(classes)
    prediction_labels = torch.from_numpy(prediction.astype(np.int64, copy=False))[None]
    target_labels = torch.from_numpy(target.astype(np.int64, copy=False))[None]
    metric.update(
        class2one_hot(prediction_labels, classes),
        class2one_hot(target_labels, classes),
        ["comparison_patient"],
    )
    _, dice = metric.compute()
    return dice[0].numpy()


def safe_mean(values: list[float]) -> float:
    defined = [float(value) for value in values if not np.isnan(value)]
    return float(np.mean(defined)) if defined else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def nifti_paths(folder: Path) -> dict[str, Path]:
    matches: dict[str, Path] = {}
    for path in sorted([*folder.glob("*.nii"), *folder.glob("*.nii.gz")]):
        patient = nifti_stem(path).casefold()
        if patient in matches:
            raise ValueError(f"Multiple NIfTI files for {patient} in {folder}")
        matches[patient] = path
    return matches


def ground_truth_paths(root: Path) -> dict[str, Path]:
    paths = sorted(root.rglob("GT.nii.gz"))
    if not paths:
        raise RuntimeError(f"No GT.nii.gz files found under {root}")
    matches: dict[str, Path] = {}
    for path in paths:
        patient = path.parent.name.casefold()
        if patient in matches:
            raise ValueError(f"Multiple ground-truth files for {patient}")
        matches[patient] = path
    return matches


def validate_labels(labels: np.ndarray, path: Path, classes: int) -> np.ndarray:
    if labels.ndim != 3:
        raise ValueError(f"Expected 3D labels at {path}, got {labels.shape}")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
        raise ValueError(f"Labels must be finite integers in {path}")
    if labels.size and (labels.min() < 0 or labels.max() >= classes):
        raise ValueError(
            f"Labels in {path} must be between 0 and {classes - 1}; "
            f"found {np.unique(labels)}"
        )
    return labels.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/heldout_predictions"),
        help="Held-out prediction root with one directory per model.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("results/heldout_predictions_postprocessed"),
        help="Existing postprocessed NIfTIs, grouped by model and patient.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/segthor_train_full/train"),
        help="Ground-truth root containing patient directories with GT.nii.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/heldout_saved_comparison"),
        help="Directory for comparison CSV summaries.",
    )
    # because computationally heavy, allow user to limit to a single patient for debugging
    parser.add_argument(
        "--patient",
        type=str,
        help="Only compare this patient ID (case-insensitive), for every model.",
    )
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--label-step", type=float, default=63.0)
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="HD95 spacing in GT array-axis order; defaults to the GT header.",
    )
    args = parser.parse_args()

    if not 2 <= args.classes <= 256:
        parser.error("--classes must be between 2 and 256")
    if not np.isfinite(args.label_step) or args.label_step <= 0:
        parser.error("--label-step must be positive and finite")
    if args.spacing is not None and (
        not np.all(np.isfinite(args.spacing)) or np.any(np.asarray(args.spacing) <= 0)
    ):
        parser.error("--spacing values must be positive and finite")
    for name, path in (
        ("--input-dir", args.input_dir),
        ("--processed-dir", args.processed_dir),
        ("--gt-dir", args.gt_dir),
    ):
        if not path.is_dir():
            parser.error(f"{name} does not exist or is not a directory: {path}")

    gt_paths = ground_truth_paths(args.gt_dir)

    selected = args.patient.casefold() if args.patient else None
    if selected is not None and selected not in gt_paths:
        parser.error(f"--patient {args.patient} not found under {args.gt_dir}")

    per_class: list[dict[str, object]] = []
    per_patient: list[dict[str, object]] = []
    model_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    models_processed = 0

    for model_dir in model_dirs:
        pngs = sorted(model_dir.glob("*.png"))
        raw_niftis = nifti_paths(model_dir)
        if pngs:
            raw_slices, _ = load_volumes(model_dir, args.classes, args.label_step)
            raw_predictions = {
                patient.casefold(): volume for patient, volume in raw_slices.items()
                if selected is None or patient.casefold() == selected
            }
            raw_affines: dict[str, np.ndarray] = {}
            raw_is_nifti = False
        elif raw_niftis:
            raw_predictions = {}
            raw_affines = {}
            for patient, path in raw_niftis.items():
                if selected is not None and patient != selected:
                    continue
                labels, image = load_nifti_labels(path, args.classes)
                raw_predictions[patient] = labels
                raw_affines[patient] = image.affine
            raw_is_nifti = True
        else:
            print(f"Skipping {model_dir.name}: no PNG or NIfTI held-out predictions")
            continue

        processed_model_dir = args.processed_dir / model_dir.name
        if not processed_model_dir.is_dir():
            raise FileNotFoundError(
                f"Processed predictions not found for {model_dir.name}: "
                f"{processed_model_dir}"
            )
        # processed_files = nifti_paths(processed_model_dir)

        processed_files = {
            p: f for p, f in nifti_paths(processed_model_dir).items()
            if selected is None or p == selected
        }
        if selected is not None and selected not in raw_predictions:
            raise ValueError(f"{model_dir.name}: no raw prediction for {args.patient}")
        
        if set(processed_files) != set(raw_predictions):
            raise ValueError(
                f"Raw and processed patient sets differ for {model_dir.name}: "
                f"raw={sorted(raw_predictions)}, processed={sorted(processed_files)}"
            )

        missing_gt = sorted(set(raw_predictions) - gt_paths.keys())
        if missing_gt:
            raise ValueError(f"{model_dir.name}: no GT found for patients {missing_gt}")

        for patient in sorted(raw_predictions):
            gt_path = gt_paths[patient]
            gt_image = nib.load(str(gt_path))
            target = validate_labels(
                np.asanyarray(gt_image.dataobj), gt_path, args.classes
            )
            patient_id = gt_path.parent.name

            raw = raw_predictions[patient]
            if raw_is_nifti:
                if raw.shape != target.shape or not np.allclose(
                    raw_affines[patient], gt_image.affine
                ):
                    raise ValueError(
                        f"Raw prediction geometry differs from GT for "
                        f"{model_dir.name}/{patient_id}"
                    )
            else:
                raw = slices_to_reference_volume(raw, target.shape)

            processed_path = processed_files[patient]
            processed, processed_image = load_nifti_labels(
                processed_path, args.classes
            )
            if processed.shape != target.shape or not np.allclose(
                processed_image.affine, gt_image.affine
            ):
                raise ValueError(
                    f"Processed prediction geometry differs from GT for "
                    f"{model_dir.name}/{patient_id}"
                )

            spacing = (
                tuple(float(value) for value in args.spacing)
                if args.spacing is not None
                else tuple(float(value) for value in gt_image.header.get_zooms()[:3])
            )
            dice_raw = patient_dice_by_class(raw, target, args.classes)
            dice_processed = patient_dice_by_class(processed, target, args.classes)
            patient_dice_raw: list[float] = []
            patient_dice_processed: list[float] = []
            patient_hd_raw: list[float] = []
            patient_hd_processed: list[float] = []

            for class_index in range(1, args.classes):
                before_dice = float(dice_raw[class_index])
                after_dice = float(dice_processed[class_index])
                before_hd = surface_hd95(
                    raw == class_index, target == class_index, spacing
                )
                after_hd = surface_hd95(
                    processed == class_index, target == class_index, spacing
                )
                per_class.append({
                    "model": model_dir.name,
                    "patient": patient_id,
                    "ground_truth": str(gt_path),
                    "class": class_index,
                    "dice_before": before_dice,
                    "dice_after": after_dice,
                    "dice_delta_after_minus_before": after_dice - before_dice,
                    "hd95_before": before_hd,
                    "hd95_after": after_hd,
                    "hd95_delta_after_minus_before": after_hd - before_hd,
                })
                if not np.isnan(before_dice):
                    patient_dice_raw.append(before_dice)
                if not np.isnan(after_dice):
                    patient_dice_processed.append(after_dice)
                if not np.isnan(before_hd):
                    patient_hd_raw.append(before_hd)
                if not np.isnan(after_hd):
                    patient_hd_processed.append(after_hd)

            before_dice_mean = safe_mean(patient_dice_raw)
            after_dice_mean = safe_mean(patient_dice_processed)
            before_hd_mean = safe_mean(patient_hd_raw)
            after_hd_mean = safe_mean(patient_hd_processed)
            per_patient.append({
                "model": model_dir.name,
                "patient": patient_id,
                "ground_truth": str(gt_path),
                "mean_foreground_dice_before": before_dice_mean,
                "mean_foreground_dice_after": after_dice_mean,
                "mean_dice_delta_after_minus_before": after_dice_mean - before_dice_mean,
                "mean_foreground_hd95_before": before_hd_mean,
                "mean_foreground_hd95_after": after_hd_mean,
                "mean_hd95_delta_after_minus_before": after_hd_mean - before_hd_mean,
            })
            print(f"Compared {model_dir.name}/{patient_id}")
        models_processed += 1

    if models_processed == 0:
        raise RuntimeError(f"No held-out predictions found under {args.input_dir}")

    model_summary: list[dict[str, object]] = []
    model_class_summary: list[dict[str, object]] = []
    for model_name in sorted({str(row["model"]) for row in per_patient}):
        model_patients = [row for row in per_patient if row["model"] == model_name]
        summary_row: dict[str, object] = {
            "model": model_name,
            "n_patients": len(model_patients),
            "mean_patient_foreground_dice_before": safe_mean([
                float(row["mean_foreground_dice_before"]) for row in model_patients
            ]),
            "mean_patient_foreground_dice_after": safe_mean([
                float(row["mean_foreground_dice_after"]) for row in model_patients
            ]),
            "mean_patient_foreground_hd95_before": safe_mean([
                float(row["mean_foreground_hd95_before"]) for row in model_patients
            ]),
            "mean_patient_foreground_hd95_after": safe_mean([
                float(row["mean_foreground_hd95_after"]) for row in model_patients
            ]),
        }
        summary_row["mean_patient_dice_delta"] = (
            summary_row["mean_patient_foreground_dice_after"]
            - summary_row["mean_patient_foreground_dice_before"]
        )
        summary_row["mean_patient_foreground_hd95_delta"] = (
            summary_row["mean_patient_foreground_hd95_after"]
            - summary_row["mean_patient_foreground_hd95_before"]
        )
        model_summary.append(summary_row)
        for class_index in range(1, args.classes):
            class_rows = [
                row for row in per_class
                if row["model"] == model_name and row["class"] == class_index
            ]
            model_class_summary.append({
                "model": model_name,
                "class": class_index,
                "n_patients_with_class": sum(
                    not np.isnan(float(row["dice_before"])) for row in class_rows
                ),
                **{
                    field: safe_mean([float(row[source]) for row in class_rows])
                    for field, source in (
                        ("mean_patient_dice_before", "dice_before"),
                        ("mean_patient_dice_after", "dice_after"),
                        ("mean_patient_hd95_before", "hd95_before"),
                        ("mean_patient_hd95_after", "hd95_after"),
                    )
                },
            })
            class_summary_row = model_class_summary[-1]
            class_summary_row["mean_patient_dice_delta"] = (
                class_summary_row["mean_patient_dice_after"]
                - class_summary_row["mean_patient_dice_before"]
            )
            class_summary_row["mean_patient_hd95_delta"] = (
                class_summary_row["mean_patient_hd95_after"]
                - class_summary_row["mean_patient_hd95_before"]
            )

    write_csv(args.output_dir / "per_patient_per_class.csv", per_class, [
        "model", "patient", "ground_truth", "class", "dice_before", "dice_after",
        "dice_delta_after_minus_before", "hd95_before", "hd95_after",
        "hd95_delta_after_minus_before",
    ])
    write_csv(args.output_dir / "per_patient_summary.csv", per_patient, [
        "model", "patient", "ground_truth", "mean_foreground_dice_before",
        "mean_foreground_dice_after", "mean_dice_delta_after_minus_before",
        "mean_foreground_hd95_before", "mean_foreground_hd95_after",
        "mean_hd95_delta_after_minus_before",
    ])
    write_csv(args.output_dir / "per_model_per_class_summary.csv", model_class_summary, [
        "model", "class", "n_patients_with_class", "mean_patient_dice_before",
        "mean_patient_dice_after", "mean_patient_dice_delta",
        "mean_patient_hd95_before", "mean_patient_hd95_after", "mean_patient_hd95_delta",
    ])
    write_csv(args.output_dir / "per_model_summary.csv", model_summary, [
        "model", "n_patients", "mean_patient_foreground_dice_before",
        "mean_patient_foreground_dice_after", "mean_patient_dice_delta",
        "mean_patient_foreground_hd95_before", "mean_patient_foreground_hd95_after",
        "mean_patient_foreground_hd95_delta",
    ])
    print(f"Saved comparison CSV summaries under {args.output_dir}")


if __name__ == "__main__":
    main()
