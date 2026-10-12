#!/usr/bin/env python3
"""Per-organ precision and recall of every model, from the per-patient metrics CSVs.

The CSVs of evaluation/evaluate_saved_predictions.py hold, per patient and organ, the Dice and the predicted and
ground-truth voxel counts. Since Dice = 2 TP / (pred + gt), the true positives are TP = Dice (pred + gt) / 2, so
    precision = TP / pred   (fraction of the predicted organ that is correct; low = over-segmentation)
    recall    = TP / gt     (fraction of the true organ that is found; low = under-segmentation)
per patient (3D), averaged like the Dice tables: cross-validation = mean ± std over the 5 fold means, held-out set =
mean ± std over its patients.

Usage: python -m evaluation.precision_recall --output results/comparison
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.compare_cv_runs import FOLDS, ORGANS, RUNS

HELDOUT = Path("results/heldout_predictions/metrics")
HELDOUT_NAMES = {"weighted_ce_25d_e50": "weighted_ce_25d"}  # cross-validation run name -> held-out metrics file


def precision_recall(rows: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """patient -> {"precision": per organ, "recall": per organ}."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        if row["organ"] not in ORGANS:
            continue
        pred, gt, dice = float(row["prediction_voxels"]), float(row["ground_truth_voxels"]), float(row["dice"])
        tp = dice * (pred + gt) / 2
        if abs(tp - round(tp)) > 1e-3 * max(tp, 1):
            raise ValueError(f"Non-integer true positives {tp} for {row['patient']} {row['organ']}")
        values = out.setdefault(row["patient"], {"precision": np.full(len(ORGANS), np.nan),
                                                 "recall": np.full(len(ORGANS), np.nan)})
        i = ORGANS.index(row["organ"])
        values["precision"][i] = tp / pred if pred else np.nan
        values["recall"][i] = tp / gt if gt else np.nan
    return out


def table(title: str, stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[str]:
    columns = [o.capitalize() for o in ORGANS] + ["Mean"]
    lines = [f"## {title}", "", "| Run | Metric | " + " | ".join(columns) + " |", "|---|---|" + "---|" * len(columns)]
    for name, (mean, std) in stats.items():
        for k, metric in enumerate(("Precision", "Recall")):
            lines.append(f"| `{name}` | {metric} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[k], std[k]))
                         + " |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    lines = ["Precision = TP / predicted voxels (low: over-segmentation), "
             "recall = TP / ground-truth voxels (low: under-segmentation), per patient in 3D.", ""]

    cv = {}
    for name, (pattern, _) in RUNS.items():
        per_fold = []
        for fold in FOLDS:
            patients = precision_recall(list(csv.DictReader(open(pattern.format(f=fold)))))
            m = np.array([[np.nanmean([p[metric] for p in patients.values()], axis=0)] for metric in ("precision", "recall")])[:, 0]
            per_fold.append(np.concatenate([m, m.mean(axis=1, keepdims=True)], axis=1))  # (2, organs + mean)
        per_fold = np.array(per_fold)
        cv[name] = (per_fold.mean(axis=0), per_fold.std(axis=0, ddof=1))
    lines += table("Cross-validation (32 patients): mean ± std over the 5 folds", cv)

    heldout = {}
    for name in RUNS:
        path = HELDOUT / f"{HELDOUT_NAMES.get(name, name)}.csv"
        if not path.is_file():
            continue
        patients = precision_recall(list(csv.DictReader(open(path))))
        values = np.array([[p[metric] for p in patients.values()] for metric in ("precision", "recall")])  # (2, pts, organs)
        values = np.concatenate([values, values.mean(axis=2, keepdims=True)], axis=2)
        heldout[name] = (values.mean(axis=1), values.std(axis=1, ddof=1))
    if heldout:
        lines += table(f"Held-out set ({len(patients)} patients): mean ± std over the patients", heldout)

    report = "\n".join(lines)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "precision_recall.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved {args.output / 'precision_recall.md'}")


if __name__ == "__main__":
    main()
