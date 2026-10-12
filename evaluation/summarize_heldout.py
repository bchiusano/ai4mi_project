#!/usr/bin/env python3
"""All metrics of every model on the held-out patients, per organ.

Reads the per-patient CSVs that evaluation/evaluate_saved_predictions.py (with --spacing-file) wrote for each
model's 5-fold ensemble on the held-out set (snellius/predict_heldout.sbatch), and tabulates per organ:
    Dice, precision, recall      (precision = TP / predicted, recall = TP / ground truth, TP from the Dice and the
                                  voxel counts: TP = Dice (pred + gt) / 2)
    HD95 and ASSD in mm, NSD     (normalized surface Dice: share of boundary within 2 mm)
as mean ± std over the patients. A complete miss (organ not predicted at all) has infinite HD95/ASSD: those are
left out of the means and counted separately.

Usage: python -m evaluation.summarize_heldout --metrics-dir results/heldout_predictions/metrics_all \
           --output results/comparison/heldout_all_metrics.md
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ORGANS = ("esophagus", "heart", "trachea", "aorta")
MODELS = {
    "ce_dice_25d": "ENet 2.5D, CE + Dice",
    "ce_dice_25d_aug": "ENet 2.5D, CE + Dice, online augmentation",
    "weighted_ce_25d": "ENet 2.5D, weighted CE (50 epochs)",
    "swin_unetr": "SwinUNETR 3D, fine-tuned from BTCV",
    "swin_unetr_eso_sampling": "SwinUNETR 3D, + esophagus-weighted sampling",
}
# metric -> (label, higher is better, decimals)
METRICS = {
    "dice": ("Dice", True, 3),
    "precision": ("Precision", True, 3),
    "recall": ("Recall", True, 3),
    "hd95_mm": ("HD95 (mm)", False, 2),
    "assd_mm": ("ASSD (mm)", False, 2),
    "nsd": ("NSD (2 mm)", True, 3),
}


def load(path: Path) -> dict[str, np.ndarray]:
    """metric -> (patients, organs)."""
    values: dict[str, dict[str, np.ndarray]] = {m: {} for m in METRICS}
    for row in csv.DictReader(open(path)):
        if row["organ"] not in ORGANS:
            continue
        i = ORGANS.index(row["organ"])
        pred, gt, dice = float(row["prediction_voxels"]), float(row["ground_truth_voxels"]), float(row["dice"])
        tp = dice * (pred + gt) / 2
        derived = {"precision": tp / pred if pred else np.nan, "recall": tp / gt if gt else np.nan}
        for metric in METRICS:
            value = derived[metric] if metric in derived else float(row[metric])
            values[metric].setdefault(row["patient"], np.full(len(ORGANS), np.nan))[i] = value
    return {m: np.array([v[p] for p in sorted(v)]) for m, v in values.items()}


def summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(organs + mean) mean, std over patients and number of complete misses, leaving out infinite values."""
    finite = np.where(np.isfinite(values), values, np.nan)
    per_patient_mean = np.nanmean(finite, axis=1, keepdims=True)  # mean over the organs per patient
    table = np.concatenate([finite, per_patient_mean], axis=1)
    return np.nanmean(table, axis=0), np.nanstd(table, axis=0, ddof=1), np.isinf(values).sum(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/heldout_predictions/metrics_all"))
    parser.add_argument("--output", type=Path, default=Path("results/comparison/heldout_all_metrics.md"))
    args = parser.parse_args()

    data = {name: load(args.metrics_dir / f"{name}.csv") for name in MODELS}
    patients = {len(next(iter(d.values()))) for d in data.values()}
    stats = {name: {m: summary(d[m]) for m in METRICS} for name, d in data.items()}
    columns = [o.capitalize() for o in ORGANS] + ["Mean"]

    def best(metric: str, column: int) -> str:
        higher = METRICS[metric][1]
        means = {name: stats[name][metric][0][column] for name in MODELS}
        return (max if higher else min)(means, key=means.get)

    lines = [f"# Held-out set: {patients.pop()} patients, 5-fold ensemble (majority vote) of every model", "",
             "Mean ± std over the patients; **bold** = best model for that organ. Dice, precision, recall and NSD: "
             "higher is better; HD95 and ASSD (mm): lower is better.", ""]

    lines += ["## Per metric", ""]
    for metric, (label, higher, digits) in METRICS.items():
        lines += [f"### {label}", "", "| Model | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for name in MODELS:
            mean, std, _ = stats[name][metric]
            cells = [f"{mean[c]:.{digits}f} ± {std[c]:.{digits}f}" for c in range(len(columns))]
            cells = [f"**{cell}**" if best(metric, c) == name else cell for c, cell in enumerate(cells)]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")
    misses = {name: stats[name]["hd95_mm"][2] for name in MODELS}
    lines += ["Complete misses (organ not predicted at all in a patient; infinite HD95/ASSD, left out of the means): "
              + ", ".join(f"`{n}` {int(m.sum())}" for n, m in misses.items()), ""]

    lines += ["## Per organ", ""]
    for c, organ in enumerate(columns):
        lines += [f"### {organ}", "", "| Model | " + " | ".join(METRICS[m][0] for m in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for name in MODELS:
            cells = []
            for metric, (_, _, digits) in METRICS.items():
                mean, std, _ = stats[name][metric]
                cell = f"{mean[c]:.{digits}f} ± {std[c]:.{digits}f}"
                cells.append(f"**{cell}**" if best(metric, c) == name else cell)
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Models: " + "; ".join(f"`{n}` = {d}" for n, d in MODELS.items()), ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
