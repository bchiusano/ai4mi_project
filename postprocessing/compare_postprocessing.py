#!/usr/bin/env python3
"""Compare saved before/after metric CSVs: tables plus per-organ and per-patient plots.

Usage:
    python -m evaluation.compare_postprocessing \
        --before-dir results/heldout_predictions/metrics \
        --after-dir results/heldout_final_metrics \
        --output-dir results/heldout_final_comparison
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
BEFORE_COLOR, AFTER_COLOR = "#4878a8", "#f28e1c"
KEYS = ["patient", "class", "organ"]
VALUE_COLUMNS = ["dice_before", "dice_after", "dice_delta", "hd95_before", "hd95_after", "hd95_delta"]


def load_pairs(before_dir: Path, after_dir: Path, models: list[str]) -> pd.DataFrame:
    frames = []
    for model in models:
        before = pd.read_csv(before_dir / f"{model}.csv")
        after = pd.read_csv(after_dir / f"{model}.csv")
        pair = before[KEYS + ["dice", "hd95_mm"]].merge(
            after[KEYS + ["dice", "hd95_mm"]], on=KEYS, suffixes=("_before", "_after"), validate="one_to_one"
        )
        if len(pair) != len(before) or len(pair) != len(after):
            raise ValueError(f"{model}: before and after CSVs do not cover the same patients/classes")
        pair.insert(0, "model", model)
        frames.append(pair)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["class"] > 0].rename(columns={"hd95_mm_before": "hd95_before", "hd95_mm_after": "hd95_after"})
    df["dice_delta"] = df["dice_after"] - df["dice_before"]
    df["hd95_delta"] = df["hd95_after"] - df["hd95_before"]
    return df


def summarise(df: pd.DataFrame, models: list[str]):
    by_organ = df.groupby(["model", "organ"])[VALUE_COLUMNS].mean().reset_index()
    per_patient = df.groupby(["model", "patient"])[VALUE_COLUMNS].mean().reset_index()  # mean over organs
    per_model = per_patient.groupby("model")[VALUE_COLUMNS].mean().reindex(models).reset_index()
    return by_organ, per_patient, per_model


def organ_bars(df: pd.DataFrame, metric: str, ylabel: str, models: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.38
    rng = np.random.default_rng(0)
    for ax, organ in zip(axes.ravel(), ORGANS):
        sub = df[df["organ"] == organ]
        for offset, suffix, color in ((-0.5, "before", BEFORE_COLOR), (0.5, "after", AFTER_COLOR)):
            column = f"{metric}_{suffix}"
            centres = x + offset * width
            means = [sub.loc[sub["model"] == model, column].mean() for model in models]
            ax.bar(centres, means, width, color=color, label=f"{suffix.capitalize()} postprocessing")
            for centre, model in zip(centres, models):
                values = sub.loc[sub["model"] == model, column].dropna().to_numpy()
                jitter = rng.uniform(-width / 4, width / 4, len(values))
                ax.scatter(centre + jitter, values, s=14, color="black", alpha=0.6, zorder=3)
        ax.set_title(organ.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"Per-organ {ylabel}: bars = mean over patients, dots = individual patients")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def patient_heatmaps(df: pd.DataFrame, title: str, models: list[str], path: Path) -> None:
    per_patient = df.groupby(["model", "patient"])[["dice_delta", "hd95_delta"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * per_patient["patient"].nunique() + 3))
    panels = (
        ("dice_delta", "RdBu", "Dice change after - before (blue = better)", "{:+.3f}"),
        ("hd95_delta", "RdBu_r", "HD95 change in mm after - before (blue = better)", "{:+.1f}"),
    )
    for ax, (column, cmap, subtitle, fmt) in zip(axes, panels):
        table = per_patient.pivot(index="patient", columns="model", values=column).reindex(columns=models)
        limit = np.nanmax(np.abs(table.to_numpy()))
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0
        image = ax.imshow(table.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_yticks(range(len(table)))
        ax.set_yticklabels(table.index)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8)
        ax.set_title(subtitle, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--before-dir", type=Path, required=True, help="Folder with <model>.csv before postprocessing.")
    parser.add_argument("--after-dir", type=Path, required=True, help="Folder with <model>.csv after postprocessing.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Default: every model with a CSV in both folders.")
    parser.add_argument("--heatmap-organs", nargs="*", default=["esophagus"],
                        help="Organs that also get their own per-patient heatmap.")
    args = parser.parse_args()

    models = args.models or sorted(
        {p.stem for p in args.before_dir.glob("*.csv")} & {p.stem for p in args.after_dir.glob("*.csv")}
    )
    if not models:
        raise RuntimeError("No model CSVs found in both --before-dir and --after-dir")

    df = load_pairs(args.before_dir, args.after_dir, models)
    by_organ, per_patient, per_model = summarise(df, models)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "all_rows.csv", index=False)
    by_organ.to_csv(args.output_dir / "summary_per_model_organ.csv", index=False)
    per_patient.to_csv(args.output_dir / "summary_per_patient.csv", index=False)
    per_model.to_csv(args.output_dir / "summary_per_model.csv", index=False)

    organ_bars(df, "dice", "Dice", models, args.output_dir / "per_organ_dice.png")
    organ_bars(df, "hd95", "HD95 (mm)", models, args.output_dir / "per_organ_hd95.png")
    patient_heatmaps(df, "Per-patient change, mean over organs", models, args.output_dir / "per_patient_all_organs.png")
    for organ in args.heatmap_organs:
        patient_heatmaps(df[df["organ"] == organ], f"Per-patient change, {organ}", models,
                         args.output_dir / f"per_patient_{organ}.png")

    pd.set_option("display.width", 200)
    print("Mean over patients and organs:")
    print(per_model.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nPer model and organ:")
    print(by_organ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved tables and plots under {args.output_dir}")


if __name__ == "__main__":
    main()