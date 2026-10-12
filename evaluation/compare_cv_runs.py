#!/usr/bin/env python3
"""Compare 5-fold cross-validation runs on the patient-level 3D Dice of their validation patients.

Every run is read from the per-patient CSVs written by evaluation/evaluate_saved_predictions.py with
--spacing-file (one per fold, see snellius/evaluate_cv_hd95.sbatch), so 2D/2.5D and 3D models (after
evaluation/nifti_predictions_to_slices.py) are scored on the same slices, with Dice and HD95 in mm.
A complete miss of an organ has an infinite HD95: HD95 averages leave those out and they are counted separately.
Each patient is in the validation set of exactly one fold, so runs can also be compared patient by patient.

Usage: python -m evaluation.compare_cv_runs --output results/comparison
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ORGANS = ("esophagus", "heart", "trachea", "aorta")
FOLDS = range(5)

# name -> (per-fold CSV path pattern, description)
RUNS = {
    "ce_dice_25d": ("results/segthor_full_preprocessed_cv/cv/ce_dice_25d/fold_{f}/best_epoch_3d_metrics.csv",
                    "ENet 2.5D, CE + Dice, 35 epochs"),
    "ce_dice_25d_aug": ("results/segthor_full_preprocessed_cv/cv/ce_dice_25d_aug/fold_{f}/best_epoch_3d_metrics.csv",
                        "ENet 2.5D, CE + Dice, online augmentation, 50 epochs"),
    "weighted_ce_25d_e50": ("results/segthor_full_preprocessed_cv/cv_e50/weighted_ce_25d/fold_{f}/best_epoch_3d_metrics.csv",
                            "ENet 2.5D, sqrt-inverse weighted CE, 50 epochs"),
    "swin_unetr": ("results/segthor_swin_unetr/cv/fold_{f}/eval_slices_3d_metrics.csv",
                   "SwinUNETR 3D, fine-tuned from BTCV"),
    "swin_unetr_eso_sampling": ("results/segthor_swin_unetr/cv_esophagus_sampling/fold_{f}/eval_slices_3d_metrics.csv",
                                "SwinUNETR 3D, fine-tuned from BTCV, esophagus-weighted patch sampling"),
}

# (baseline, candidate, question)
COMPARISONS = [
    ("ce_dice_25d", "ce_dice_25d_aug", "Does online augmentation help CE + Dice?"),
    ("weighted_ce_25d_e50", "ce_dice_25d", "CE + Dice vs weighted CE"),
    ("ce_dice_25d", "swin_unetr", "3D SwinUNETR vs the 2.5D baseline"),
    ("ce_dice_25d_aug", "swin_unetr", "3D SwinUNETR vs the augmented 2.5D model"),
    ("swin_unetr", "swin_unetr_eso_sampling", "Does esophagus-weighted patch sampling help SwinUNETR?"),
]


METRICS = {"dice": ("3D Dice", "higher"), "hd95_mm": ("3D HD95 (mm)", "lower")}


def load_run(pattern: str) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """metric -> fold -> patient -> value per organ (esophagus, heart, trachea, aorta)."""
    run: dict[str, dict[int, dict[str, np.ndarray]]] = {metric: {} for metric in METRICS}
    for fold in FOLDS:
        path = Path(pattern.format(f=fold))
        if not path.is_file():
            raise FileNotFoundError(path)
        for metric in METRICS:
            run[metric][fold] = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                if row["organ"] not in ORGANS:
                    continue
                for metric in METRICS:
                    values = run[metric][fold].setdefault(row["patient"], np.full(len(ORGANS), np.nan))
                    values[ORGANS.index(row["organ"])] = float(row[metric])
    return run


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean that leaves out complete misses (infinite HD95) and absent organs (NaN)."""
    finite = np.where(np.isfinite(values), values, np.nan)
    return np.nanmean(finite, axis=axis)


def fold_scores(per_fold: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    """(folds, organs + mean): per fold, the mean over its patients, like the training summaries."""
    scores = []
    for fold in FOLDS:
        organs = finite_mean(np.stack(list(per_fold[fold].values())), axis=0)
        scores.append(np.append(organs, organs.mean()))
    return np.array(scores)


def patient_scores(per_fold: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """patient -> value per organ + mean over the organs (infinite if any organ was missed)."""
    return {patient: np.append(values, np.mean(values))
            for fold in FOLDS for patient, values in per_fold[fold].items()}


def misses(per_fold: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    """Number of complete misses (infinite HD95) per organ."""
    return np.isinf(np.stack([v for fold in FOLDS for v in per_fold[fold].values()])).sum(axis=0)


def fmt(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = {name: load_run(pattern) for name, (pattern, _) in RUNS.items()}
    folds = {m: {name: fold_scores(run[m]) for name, run in runs.items()} for m in METRICS}
    patients = {m: {name: patient_scores(run[m]) for name, run in runs.items()} for m in METRICS}
    columns = [o.capitalize() for o in ORGANS] + ["Mean"]
    lines = []

    for metric, (label, direction) in METRICS.items():
        digits = 3 if metric == "dice" else 2
        lines += [f"## Overall: {label} on the validation patients, mean ± std over the 5 folds ({direction} is better)",
                  "", "| Run | Description | " + " | ".join(columns) + " |", "|---|---|" + "---|" * len(columns)]
        for name, (_, description) in RUNS.items():
            s = folds[metric][name]
            lines.append(f"| `{name}` | {description} | "
                         + " | ".join(fmt(s[:, i].mean(), s[:, i].std(ddof=1), digits) for i in range(len(columns)))
                         + " |")
        if metric == "hd95_mm":
            lines += ["", "Complete misses (organ not predicted at all in a patient, infinite HD95, left out above):", "",
                      "| Run | " + " | ".join(columns[:-1]) + " |", "|---|" + "---|" * len(ORGANS)]
            for name in RUNS:
                lines.append(f"| `{name}` | " + " | ".join(str(n) for n in misses(runs[name][metric])) + " |")
        lines += ["", f"## Mean foreground {label} per fold", "",
                  "| Run | " + " | ".join(f"Fold {f}" for f in FOLDS) + " |", "|---|" + "---|" * len(FOLDS)]
        for name in RUNS:
            lines.append(f"| `{name}` | " + " | ".join(f"{v:.{digits}f}" for v in folds[metric][name][:, -1]) + " |")
        lines.append("")

    for base, cand, question in COMPARISONS:
        lines += [f"## {question}", "", f"`{cand}` minus `{base}`.", "",
                  "| | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for metric, (label, direction) in METRICS.items():
            sign = 1 if direction == "higher" else -1
            digits = 3 if metric == "dice" else 2
            delta = folds[metric][cand] - folds[metric][base]
            shared = sorted(patients[metric][base].keys() & patients[metric][cand].keys())
            a = np.array([patients[metric][base][p] for p in shared])
            b = np.array([patients[metric][cand][p] for p in shared])
            better = sign * (b - a) > 0
            p_values = []
            for i in range(len(columns)):
                both = np.isfinite(a[:, i]) & np.isfinite(b[:, i])  # Wilcoxon needs finite pairs
                p_values.append(wilcoxon(b[both, i], a[both, i]).pvalue)
            lines += [f"| {label}: mean difference over folds | " + " | ".join(f"{d:+.{digits}f}" for d in delta.mean(0)) + " |",
                      f"| {label}: folds where `{cand}` is better | "
                      + " | ".join(f"{int((sign * delta[:, i] > 0).sum())}/5" for i in range(len(columns))) + " |",
                      f"| {label}: patients where `{cand}` is better (of {len(shared)}) | "
                      + " | ".join(str(int(better[:, i].sum())) for i in range(len(columns))) + " |",
                      f"| {label}: Wilcoxon p (paired over patients) | " + " | ".join(f"{p:.3f}" for p in p_values) + " |"]
        dice_delta = folds["dice"][cand] - folds["dice"][base]
        lines += ["", "Per fold (mean foreground Dice): " + ", ".join(
            f"fold {f}: {folds['dice'][base][f, -1]:.3f} → {folds['dice'][cand][f, -1]:.3f} ({dice_delta[f, -1]:+.3f})"
            for f in FOLDS), ""]

    report = "\n".join(lines)
    (args.output / "cv_comparison.md").write_text(report, encoding="utf-8")

    with (args.output / "cv_fold_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "fold", "metric"] + [c.lower() for c in columns])
        for metric in METRICS:
            for name in RUNS:
                for fold in FOLDS:
                    writer.writerow([name, fold, metric] + [f"{v:.6f}" for v in folds[metric][name][fold]])

    print(report)
    print(f"Saved {args.output / 'cv_comparison.md'} and {args.output / 'cv_fold_scores.csv'}")


if __name__ == "__main__":
    main()
