#!/usr/bin/env python3
"""Postprocessing effect on the held-out set, with the same metrics as evaluation/summarize_heldout.py.

Writes under --output-dir:
  table1_mean_per_model.csv            mean change per model (mean over patients and foreground organs)
  table2_<model>_top3_patients.csv     the 3 patients with the largest Dice increase for --top-model
  table3_swin_models_per_organ.csv     mean change per organ for the --swin-models
  after_table_<model>.csv              "mean ± std" per organ in the layout of the summarize_heldout tables,
                                       ready for plotting.render_results_table
                                       (--check-before also writes before_table_<model>.csv)
Changes are after - before: positive Dice / precision / recall / NSD and negative HD95 / ASSD are better.
Dice, precision, recall, HD95, ASSD and NSD come from evaluation.summarize_heldout (same code as the
before-postprocessing table). A complete miss (infinite HD95/ASSD) is left out of the means.

Usage: python -m evaluation.postprocessing_tables
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.summarize_heldout import METRICS, MODELS, ORGANS, load, summary

DELTA_HEADERS = {
    "dice": "Δ Dice ↑",
    "precision": "Δ Precision ↑",
    "recall": "Δ Recall ↑",
    "hd95_mm": "Δ HD95 (mm) ↓",
    "assd_mm": "Δ ASSD (mm) ↓",
    "nsd": "Δ NSD (2 mm) ↑",
}
DELTA_COLS = list(DELTA_HEADERS.values())
REQUIRED_COLUMNS = {"prediction_voxels", "ground_truth_voxels", "hd95_mm", "assd_mm", "nsd"}
KEYS = ["model", "patient", "organ"]


def check_columns(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise ValueError(
            f"{path} has no column(s) {sorted(missing)}: re-run the evaluation with the current "
            "evaluation/evaluate_saved_predictions.py (and --spacing-file)"
        )


def long_table(directory: Path, model: str) -> pd.DataFrame:
    """One row per patient and organ with all six metrics (infinite values become NaN)."""
    path = directory / f"{model}.csv"
    check_columns(path)
    with path.open(newline="", encoding="utf-8") as handle:
        patients = sorted({row["patient"] for row in csv.DictReader(handle)})
    values = load(path)  # metric -> (patients, organs), patients sorted
    rows = [
        {"model": model, "patient": patient, "organ": organ,
         **{metric: values[metric][p, o] for metric in METRICS}}
        for p, patient in enumerate(patients)
        for o, organ in enumerate(ORGANS)
    ]
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def delta_table(before_dir: Path, after_dir: Path, model: str) -> pd.DataFrame:
    before, after = long_table(before_dir, model), long_table(after_dir, model)
    merged = before.merge(after, on=KEYS, suffixes=("_before", "_after"), validate="one_to_one")
    if not len(merged) == len(before) == len(after):
        raise ValueError(f"{model}: before and after CSVs do not cover the same patients and organs")
    for metric, header in DELTA_HEADERS.items():
        merged[header] = merged[f"{metric}_after"] - merged[f"{metric}_before"]
    return merged[KEYS + DELTA_COLS]


def mean_std_table(directory: Path, model: str) -> tuple[pd.DataFrame, int]:
    """Same layout as the before-postprocessing table: organs + Mean, 'mean ± std' over patients."""
    path = directory / f"{model}.csv"
    check_columns(path)
    values = load(path)
    stats = {metric: summary(values[metric]) for metric in METRICS}
    labels = [organ.capitalize() for organ in ORGANS] + ["Mean"]
    rows = []
    for c, label in enumerate(labels):
        row = {"Organ": label}
        for metric, (name, higher, digits) in METRICS.items():
            mean, std, _ = stats[metric]
            row[f"{name} {'↑' if higher else '↓'}"] = f"{mean[c]:.{digits}f} ± {std[c]:.{digits}f}"
        rows.append(row)
    return pd.DataFrame(rows), int(stats["hd95_mm"][2].sum())


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--before-dir", type=Path, default=Path("results/heldout_predictions/metrics_all"))
    parser.add_argument("--after-dir", type=Path, default=Path("results/heldout_final_metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/heldout_final_comparison"))
    parser.add_argument("--top-model", default="swin_unetr", help="Model for the top-3 patients table.")
    parser.add_argument("--swin-models", nargs="+", default=["swin_unetr_eso_sampling", "swin_unetr"],
                        help="Models for the per-organ delta table and the mean ± std tables.")
    parser.add_argument("--check-before", action="store_true",
                        help="Also write the before-postprocessing mean ± std tables (to compare with your existing table).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in MODELS if (args.before_dir / f"{m}.csv").is_file() and (args.after_dir / f"{m}.csv").is_file()]
    skipped = sorted(set(MODELS) - set(models))
    print(f"Models compared: {models}" + (f" | skipped (CSV missing before or after): {skipped}" if skipped else ""))
    for needed in [args.top_model, *args.swin_models]:
        if needed not in models:
            raise ValueError(f"{needed} needs a CSV in both {args.before_dir} and {args.after_dir}")

    deltas = pd.concat([delta_table(args.before_dir, args.after_dir, m) for m in models], ignore_index=True)
    per_patient = deltas.groupby(["model", "patient"])[DELTA_COLS].mean().reset_index()  # mean over organs

    # Table 1: mean change per model
    table1 = (per_patient.groupby("model")[DELTA_COLS].mean().reindex(models)
              .round(4).reset_index().rename(columns={"model": "Model"}))

    # Table 2: top 3 patients that improved in Dice for one model
    dice_col = DELTA_HEADERS["dice"]
    one = per_patient[per_patient["model"] == args.top_model]
    improved = one[one[dice_col] > 0]
    table2 = (improved.nlargest(3, dice_col)[["patient"] + DELTA_COLS]
              .round(4).reset_index(drop=True).rename(columns={"patient": "Patient"}))

    # Table 3: mean change per organ for the Swin models
    sub = deltas[deltas["model"].isin(args.swin_models)].copy()
    sub["organ"] = pd.Categorical(sub["organ"], ORGANS, ordered=True)
    table3 = (sub.groupby(["model", "organ"], observed=True)[DELTA_COLS].mean()
              .round(4).reset_index().rename(columns={"model": "Model", "organ": "Organ"}))
    table3["Organ"] = table3["Organ"].astype(str).str.capitalize()

    show("Table 1: mean change per model (after - before, mean over patients and organs)", table1)
    show(f"Table 2: {args.top_model}, top 3 patients by Dice change "
         f"({len(improved)} of {len(one)} patients improved in Dice)", table2)
    show("Table 3: mean change per organ", table3)
    table1.to_csv(args.output_dir / "table1_mean_per_model.csv", index=False, encoding="utf-8")
    table2.to_csv(args.output_dir / f"table2_{args.top_model}_top3_patients.csv", index=False, encoding="utf-8")
    table3.to_csv(args.output_dir / "table3_swin_models_per_organ.csv", index=False, encoding="utf-8")

    # mean ± std tables after postprocessing (--check-before adds the before tables as a sanity check)
    tags = [("after", args.after_dir)] + ([("before", args.before_dir)] if args.check_before else [])
    for model in args.swin_models:
        for tag, directory in tags:
            table, misses = mean_std_table(directory, model)
            table.to_csv(args.output_dir / f"{tag}_table_{model}.csv", index=False, encoding="utf-8")
            show(f"{model}, {tag} postprocessing (complete misses left out of HD95/ASSD means: {misses})", table)
    print(f"\nSaved CSVs under {args.output_dir}")


if __name__ == "__main__":
    main()