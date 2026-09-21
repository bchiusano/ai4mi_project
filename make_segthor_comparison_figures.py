#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("CE", "Dice", "CE + Dice")
COLORS = {"CE": "#4C78A8", "Dice": "#F58518", "CE + Dice": "#54A24B"}
ORGANS = ("Esophagus", "Heart", "Trachea", "Aorta")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def detailed_scores(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_csv(path)
    patients = sorted({row["patient"] for row in rows})
    scores = np.full((len(patients), 4), np.nan, dtype=np.float64)
    patient_index = {patient: index for index, patient in enumerate(patients)}
    for row in rows:
        class_index = int(row["class"])
        if 1 <= class_index <= 4:
            scores[patient_index[row["patient"]], class_index - 1] = float(row["dice"])
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Incomplete foreground scores in {path}")
    return patients, scores


def label_bars(axis: plt.Axes, bars) -> None:
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create presentation figures for SegTHOR loss experiments")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ce-baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_paths = {
        "CE": args.results_root / "ce_all_epochs" / "baseline_ce_best_epoch_3d_dice.csv",
        "Dice": args.results_root / "dice" / "best_epoch_3d_dice.csv",
        "CE + Dice": args.results_root / "ce_dice" / "best_epoch_3d_dice.csv",
    }
    method_scores: dict[str, np.ndarray] = {}
    patient_ids: list[str] | None = None
    for method, path in detailed_paths.items():
        patients, scores = detailed_scores(path)
        if patient_ids is None:
            patient_ids = patients
        elif patients != patient_ids:
            raise RuntimeError(f"Patient mismatch for {method}: {patients} != {patient_ids}")
        method_scores[method] = scores

    # Curves and selected epochs use exactly the same patient-level 3D criterion.
    ce_epoch_rows = read_csv(
        args.results_root / "ce_all_epochs" / "baseline_ce_3d_dice_by_epoch.csv"
    )
    curves = {
        "CE": np.asarray([float(row["mean_foreground_3d_dice"]) for row in ce_epoch_rows]),
        "Dice": np.nanmean(
            np.load(args.results_root / "dice" / "dice3d_val.npy")[:, :, 1:], axis=(1, 2)
        ),
        "CE + Dice": np.nanmean(
            np.load(args.results_root / "ce_dice" / "dice3d_val.npy")[:, :, 1:], axis=(1, 2)
        ),
    }
    best_epochs = {method: int(np.nanargmax(curve)) for method, curve in curves.items()}
    overall = {method: float(np.mean(method_scores[method])) for method in METHODS}
    organ_scores = {
        method: np.mean(method_scores[method], axis=0) for method in METHODS
    }

    slice_arrays = {
        "CE": np.load(args.ce_baseline_dir / "dice_val.npy"),
        "Dice": np.load(args.results_root / "dice" / "dice_val.npy"),
        "CE + Dice": np.load(args.results_root / "ce_dice" / "dice_val.npy"),
    }
    slice_scores = {
        method: float(np.mean(slice_arrays[method][best_epochs[method], :, 1:]))
        for method in METHODS
    }

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        summary_rows.append({
            "method": method,
            "best_epoch": best_epochs[method],
            "mean_foreground_3d_dice": overall[method],
            "mean_foreground_slice_dice": slice_scores[method],
            **{
                f"{organ.lower()}_3d_dice": organ_scores[method][index]
                for index, organ in enumerate(ORGANS)
            },
        })
    write_csv(args.output_dir / "loss_comparison_summary.csv", summary_rows)

    patient_rows: list[dict[str, object]] = []
    assert patient_ids is not None
    for method in METHODS:
        for patient_index, patient in enumerate(patient_ids):
            row: dict[str, object] = {
                "method": method,
                "patient": patient,
                "mean_foreground_3d_dice": float(np.mean(method_scores[method][patient_index])),
            }
            row.update({
                f"{organ.lower()}_3d_dice": method_scores[method][patient_index, organ_index]
                for organ_index, organ in enumerate(ORGANS)
            })
            patient_rows.append(row)
    write_csv(args.output_dir / "per_patient_3d_dice.csv", patient_rows)

    # Overall method comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(METHODS, [overall[m] for m in METHODS], color=[COLORS[m] for m in METHODS])
    label_bars(ax, bars)
    ax.set_ylabel("Mean foreground patient-level 3D Dice")
    ax.set_ylim(0, 0.7)
    ax.set_title("Loss-function comparison on corrected SegTHOR validation data")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "overall_3d_dice.png", dpi=200)
    plt.close(fig)

    # Per-organ grouped comparison.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(ORGANS))
    width = 0.24
    for index, method in enumerate(METHODS):
        ax.bar(x + (index - 1) * width, organ_scores[method], width,
               label=method, color=COLORS[method])
    ax.set_xticks(x, ORGANS)
    ax.set_ylabel("Patient-level 3D Dice")
    ax.set_ylim(0, 0.9)
    ax.set_title("Per-organ validation performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_organ_3d_dice.png", dpi=200)
    plt.close(fig)

    # Patient scatter shows the variation hidden by the overall mean.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method_index, method in enumerate(METHODS):
        patient_means = np.mean(method_scores[method], axis=1)
        offsets = np.linspace(-0.09, 0.09, len(patient_means))
        ax.scatter(
            np.full(len(patient_means), method_index) + offsets,
            patient_means,
            s=65,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.hlines(overall[method], method_index - 0.22, method_index + 0.22,
                  color="black", linewidth=2.2)
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_ylabel("Mean foreground patient-level 3D Dice")
    ax.set_ylim(0.35, 0.78)
    ax.set_title("Validation performance across five patients")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.99, 0.02, "Dots: patients   Black line: mean", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_patient_3d_dice_scatter.png", dpi=200)
    plt.close(fig)

    # Comparable 3D validation curves.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHODS:
        epochs = np.arange(len(curves[method]))
        ax.plot(epochs, curves[method], linewidth=2, label=method, color=COLORS[method])
        epoch = best_epochs[method]
        ax.scatter(epoch, curves[method][epoch], color=COLORS[method], edgecolor="black", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean foreground patient-level 3D Dice")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.7)
    ax.set_title("Validation 3D Dice during training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_3d_dice_curves.png", dpi=200)
    plt.close(fig)

    # Show why the old per-slice metric and the volume metric are not interchangeable.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(METHODS))
    width = 0.34
    old_bars = ax.bar(x - width / 2, [slice_scores[m] for m in METHODS], width,
                      label="Mean 2D slice Dice", color="#B8B8B8")
    volume_bars = ax.bar(x + width / 2, [overall[m] for m in METHODS], width,
                         label="Patient-level 3D Dice", color=[COLORS[m] for m in METHODS])
    label_bars(ax, old_bars)
    label_bars(ax, volume_bars)
    ax.set_xticks(x, METHODS)
    ax.set_ylabel("Mean foreground Dice")
    ax.set_ylim(0, 0.82)
    ax.set_title("Slice-averaged and patient-volume Dice answer different questions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "slice_vs_volume_dice.png", dpi=200)
    plt.close(fig)

    markdown = [
        "# SegTHOR loss comparison",
        "",
        "All scores are from the same five-patient corrected validation split. The primary metric is the macro mean of patient-level 3D Dice over the four foreground organs.",
        "",
        "| Loss | Best epoch | Mean 3D Dice | Esophagus | Heart | Trachea | Aorta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['method']} | {row['best_epoch']} | "
            f"{row['mean_foreground_3d_dice']:.3f} | {row['esophagus_3d_dice']:.3f} | "
            f"{row['heart_3d_dice']:.3f} | {row['trachea_3d_dice']:.3f} | "
            f"{row['aorta_3d_dice']:.3f} |"
        )
    markdown.extend([
        "",
        "CE + Dice achieved the highest overall score. Dice alone achieved the highest esophagus and aorta scores, while CE + Dice achieved the highest heart and trachea scores.",
        "",
        "Validation losses are not compared across methods because each objective has a different numerical scale.",
    ])
    (args.output_dir / "RESULTS.md").write_text("\n".join(markdown) + "\n")

    print(f"Saved comparison outputs to: {args.output_dir}")
    for row in summary_rows:
        print(f"{row['method']}: epoch {row['best_epoch']}, 3D Dice={row['mean_foreground_3d_dice']:.6f}")


if __name__ == "__main__":
    main()
