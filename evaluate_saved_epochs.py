#!/usr/bin/env python3

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_saved_predictions import (
    ORGAN_NAMES,
    evaluate_prediction_directory,
)


EPOCH_PATTERN = re.compile(r"^iter(\d+)$")


def evaluate_epoch(
    task: tuple[int, Path, Path, int, float],
) -> tuple[int, list[str], np.ndarray, list[dict[str, object]]]:
    epoch, pred_dir, gt_dir, classes, label_step = task
    patients, dice, detailed_rows = evaluate_prediction_directory(
        pred_dir, gt_dir, classes, label_step
    )
    return epoch, patients, dice, detailed_rows


def discover_epochs(run_dir: Path, split: str) -> list[tuple[int, Path]]:
    epochs: list[tuple[int, Path]] = []
    for candidate in run_dir.glob("iter*"):
        match = EPOCH_PATTERN.fullmatch(candidate.name)
        prediction_dir = candidate / split
        if match and prediction_dir.is_dir():
            epochs.append((int(match.group(1)), prediction_dir))
    epochs.sort()
    if not epochs:
        raise RuntimeError(f"No iter###/{split} prediction folders found in {run_dir}")
    return epochs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved iter### prediction folder using patient-level "
            "3D Dice and select the best epoch by mean foreground Dice."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="all_epochs")
    parser.add_argument("--split", default="val")
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--label-step", type=float, default=63.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Epochs to evaluate concurrently (use CPUs allocated by Slurm).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    epochs = discover_epochs(args.run_dir, args.split)
    expected_epoch_numbers = list(range(epochs[0][0], epochs[-1][0] + 1))
    actual_epoch_numbers = [epoch for epoch, _ in epochs]
    if actual_epoch_numbers != expected_epoch_numbers:
        raise RuntimeError(
            f"Epoch folders are not contiguous: found {actual_epoch_numbers}"
        )

    all_dice: list[np.ndarray] = []
    epoch_rows: list[dict[str, object]] = []
    detailed_rows_by_epoch: dict[int, list[dict[str, object]]] = {}
    expected_patients: list[str] | None = None

    tasks = [
        (epoch, pred_dir, args.gt_dir, args.classes, args.label_step)
        for epoch, pred_dir in epochs
    ]
    if args.workers == 1:
        results = map(evaluate_epoch, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(evaluate_epoch, tasks)

    try:
        for position, (epoch, patients, dice, detailed_rows) in enumerate(results, start=1):
            if expected_patients is None:
                expected_patients = patients
            elif patients != expected_patients:
                raise RuntimeError(
                    f"Patients changed at epoch {epoch}: {patients} != {expected_patients}"
                )

            all_dice.append(dice)
            detailed_rows_by_epoch[epoch] = detailed_rows
            row: dict[str, object] = {
                "epoch": epoch,
                "mean_foreground_3d_dice": float(np.nanmean(dice[:, 1:])),
            }
            for class_index in range(1, args.classes):
                organ = (ORGAN_NAMES[class_index]
                         if class_index < len(ORGAN_NAMES)
                         else f"class_{class_index}")
                row[f"{organ}_3d_dice"] = float(np.nanmean(dice[:, class_index]))
            epoch_rows.append(row)
            print(
                f"[{position:02d}/{len(epochs):02d}] epoch={epoch:03d} "
                f"mean_foreground_3d_dice={row['mean_foreground_3d_dice']:.6f}",
                flush=True,
            )
    finally:
        if args.workers != 1:
            executor.shutdown()

    dice_array = np.stack(all_dice)
    scores = np.asarray(
        [row["mean_foreground_3d_dice"] for row in epoch_rows], dtype=np.float64
    )
    best_position = int(np.nanargmax(scores))
    best_epoch = epochs[best_position][0]
    best_score = scores[best_position]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix
    write_csv(args.output_dir / f"{stem}_3d_dice_by_epoch.csv", epoch_rows)
    write_csv(
        args.output_dir / f"{stem}_best_epoch_3d_dice.csv",
        detailed_rows_by_epoch[best_epoch],
    )
    np.save(args.output_dir / f"{stem}_3d_dice_by_epoch.npy", dice_array)
    (args.output_dir / f"{stem}_patients.txt").write_text(
        "\n".join(expected_patients or []) + "\n"
    )

    summary_lines = [
        f"Run directory: {args.run_dir}",
        f"Evaluated epochs: {len(epochs)} ({epochs[0][0]}-{epochs[-1][0]})",
        f"Best patient-level 3D-Dice epoch: {best_epoch}",
        f"Mean foreground patient-level 3D Dice: {best_score:.6f}",
    ]
    best_dice = dice_array[best_position]
    for class_index in range(1, args.classes):
        organ = (ORGAN_NAMES[class_index]
                 if class_index < len(ORGAN_NAMES)
                 else f"class_{class_index}")
        summary_lines.append(
            f"{organ.capitalize()} 3D Dice: "
            f"{np.nanmean(best_dice[:, class_index]):.6f}"
        )
    summary = "\n".join(summary_lines) + "\n"
    (args.output_dir / f"{stem}_best_epoch_3d_dice.txt").write_text(summary)

    epoch_numbers = np.asarray(actual_epoch_numbers)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_numbers, scores, marker="o", linewidth=2, label="Mean foreground")
    for class_index in range(1, args.classes):
        organ = (ORGAN_NAMES[class_index]
                 if class_index < len(ORGAN_NAMES)
                 else f"class {class_index}")
        plt.plot(
            epoch_numbers,
            np.nanmean(dice_array[:, :, class_index], axis=1),
            linewidth=1.2,
            alpha=0.8,
            label=organ.capitalize(),
        )
    plt.scatter([best_epoch], [best_score], color="black", zorder=5)
    plt.annotate(
        f"best: epoch {best_epoch} ({best_score:.3f})",
        (best_epoch, best_score),
        xytext=(8, 8),
        textcoords="offset points",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Patient-level 3D Dice")
    plt.title("Validation 3D Dice by epoch")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"{stem}_3d_dice_by_epoch.png", dpi=180)
    plt.close()

    print("\n" + summary, end="")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
