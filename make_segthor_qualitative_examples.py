#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


METHODS = ("CE", "Dice", "CE + Dice")
CLASS_NAMES = {1: "Esophagus", 2: "Heart", 3: "Trachea", 4: "Aorta"}
CLASS_COLORS = {
    1: (0.89, 0.10, 0.47, 0.62),
    2: (1.00, 0.72, 0.10, 0.62),
    3: (0.13, 0.72, 0.88, 0.62),
    4: (0.24, 0.70, 0.35, 0.62),
}


def load_labels(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path))
    labels = np.rint(values.astype(np.float64) / 63.0).astype(np.uint8)
    if not set(np.unique(labels)) <= set(range(5)):
        raise ValueError(f"Unexpected labels in {path}: {np.unique(values)}")
    return labels


def foreground_slice_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    scores: list[float] = []
    for class_index in range(1, 5):
        pred_class = prediction == class_index
        target_class = target == class_index
        denominator = np.count_nonzero(pred_class) + np.count_nonzero(target_class)
        if denominator:
            scores.append(
                2 * np.count_nonzero(pred_class & target_class) / denominator
            )
    return float(np.mean(scores)) if scores else float("nan")


def overlay(axis: plt.Axes, image: np.ndarray, labels: np.ndarray) -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for class_index, color in CLASS_COLORS.items():
        rgba[labels == class_index] = color
    axis.imshow(rgba)
    axis.axis("off")


def colored_mask(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
    for class_index, color in CLASS_COLORS.items():
        rgb[labels == class_index] = color[:3]
    return rgb


def choose_slice(
    patient: str,
    gt_dir: Path,
    prediction_dirs: dict[str, Path],
) -> tuple[str, dict[str, float | int]]:
    candidates = sorted(gt_dir.glob(f"{patient}_*.png"))
    if not candidates:
        raise RuntimeError(f"No ground-truth slices found for {patient}")

    ranked: list[tuple[tuple[float, ...], str, dict[str, float | int]]] = []
    for gt_path in candidates:
        if not all((directory / gt_path.name).is_file() for directory in prediction_dirs.values()):
            continue
        target = load_labels(gt_path)
        foreground_voxels = int(np.count_nonzero(target))
        organs_present = int(len(set(np.unique(target)) - {0}))
        predictions = {
            method: load_labels(directory / gt_path.name)
            for method, directory in prediction_dirs.items()
        }
        total_error = sum(np.count_nonzero(prediction != target) for prediction in predictions.values())
        disagreement = sum(
            np.count_nonzero(predictions[left] != predictions[right])
            for left, right in (("CE", "Dice"), ("CE", "CE + Dice"), ("Dice", "CE + Dice"))
        )
        details: dict[str, float | int] = {
            "organs_present": organs_present,
            "ground_truth_foreground_pixels": foreground_voxels,
            "total_prediction_error_pixels": int(total_error),
            "prediction_disagreement_pixels": int(disagreement),
        }
        for method, prediction in predictions.items():
            details[f"{method}_slice_dice"] = foreground_slice_dice(prediction, target)

        # Prefer anatomically informative slices containing more organs, then
        # slices where the methods visibly differ, then larger foreground area.
        rank = (float(organs_present), float(disagreement), float(total_error), float(foreground_voxels))
        ranked.append((rank, gt_path.name, details))

    if not ranked:
        raise RuntimeError(f"No complete prediction set found for {patient}")
    _, filename, details = max(ranked, key=lambda item: item[0])
    return filename, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qualitative SegTHOR prediction comparisons")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--ce-dir", type=Path, required=True)
    parser.add_argument("--dice-dir", type=Path, required=True)
    parser.add_argument("--ce-dice-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Patient_01", "Patient_15", "Patient_19"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_dirs = {
        "CE": args.ce_dir,
        "Dice": args.dice_dir,
        "CE + Dice": args.ce_dice_dir,
    }
    for directory in [args.image_dir, args.gt_dir, *prediction_dirs.values()]:
        count = len(list(directory.glob("*.png")))
        if count != 915:
            raise RuntimeError(f"Expected 915 PNG files in {directory}, found {count}")

    selections: list[dict[str, object]] = []
    for patient in args.patients:
        filename, details = choose_slice(patient, args.gt_dir, prediction_dirs)
        selections.append({"patient": patient, "filename": filename, **details})

    columns = ("CT + ground truth", "Ground truth", *METHODS)
    fig, axes = plt.subplots(
        len(selections), len(columns),
        figsize=(15, 3.15 * len(selections)),
        squeeze=False,
    )
    for row_index, selection in enumerate(selections):
        filename = str(selection["filename"])
        image = np.asarray(Image.open(args.image_dir / filename))
        target = load_labels(args.gt_dir / filename)

        overlay(axes[row_index, 0], image, target)
        axes[row_index, 1].imshow(colored_mask(target))
        axes[row_index, 1].axis("off")
        for column_index, method in enumerate(METHODS, start=2):
            prediction = load_labels(prediction_dirs[method] / filename)
            overlay(axes[row_index, column_index], image, prediction)

        axes[row_index, 0].text(
            -0.05,
            0.5,
            f"{selection['patient']}\nslice {Path(filename).stem.split('_')[-1]}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
        )
        for column_index, title in enumerate(columns):
            if row_index == 0:
                axes[row_index, column_index].set_title(title, fontsize=12)

    legend = [
        Patch(facecolor=CLASS_COLORS[index], edgecolor="none", label=name)
        for index, name in CLASS_NAMES.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Qualitative validation examples", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    fig.savefig(args.output_dir / "qualitative_examples.png", dpi=200)
    plt.close(fig)

    csv_path = args.output_dir / "qualitative_examples.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=selections[0].keys())
        writer.writeheader()
        writer.writerows(selections)

    print(f"Saved qualitative figure to: {args.output_dir / 'qualitative_examples.png'}")
    for selection in selections:
        print(f"{selection['patient']}: {selection['filename']}")


if __name__ == "__main__":
    main()
