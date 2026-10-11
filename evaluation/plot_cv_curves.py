#!/usr/bin/env python3
"""Training curves of the cross-validation runs: train/val loss and validation 3D Dice over the epochs.

One figure per run, one column per fold (shared axes):
  top:    train loss (solid) and validation loss (dashed), mean per epoch, log scale
  bottom: validation patient-level 3D Dice per organ and their mean, best epoch marked
Plus an overview with the mean validation 3D Dice of every 2.5D run (mean over the folds still training,
shown while at least MIN_FOLDS of them are).

The 2.5D runs (main.py) log everything per epoch. The MONAI SwinUNETR bundle logs the train loss per iteration and
the mean validation Dice every 10 epochs in its own resampled 1.5 mm space, and, from the esophagus sampling
run on, the validation loss (the training DiceCE loss on each whole validation volume).

Usage: python -m evaluation.plot_cv_curves --output results/comparison/curves
"""

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDS = range(5)
ORGANS = ("Esophagus", "Heart", "Trachea", "Aorta")

# Validated categorical palette (all pairs pass the CVD and normal-vision checks, see the dataviz skill)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
ORGAN_COLOURS = (BLUE, ORANGE, AQUA, VIOLET)
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

RUNS_25D = {
    "ce_dice_25d": ("results/segthor_full_preprocessed_cv/cv/ce_dice_25d", "ENet 2.5D, CE + Dice"),
    "ce_dice_25d_aug": ("results/segthor_full_preprocessed_cv/cv/ce_dice_25d_aug",
                        "ENet 2.5D, CE + Dice, online augmentation"),
    "weighted_ce_25d_e50": ("results/segthor_full_preprocessed_cv/cv_e50/weighted_ce_25d",
                            "ENet 2.5D, sqrt-inverse weighted CE"),
}
SWIN_RUNS = {  # runs without results yet are skipped
    "swin_unetr": ("results/segthor_swin_unetr/cv", "SwinUNETR 3D, fine-tuned from BTCV"),
    "swin_unetr_esophagus_sampling": ("results/segthor_swin_unetr/cv_esophagus_sampling",
                                      "SwinUNETR 3D, fine-tuned from BTCV, esophagus-weighted patch sampling"),
}


def log_ticks(ax) -> None:
    """Label 1, 2 and 5 x 10^n: readable whether the loss spans less or more than a decade."""
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def load_25d(fold_dir: Path) -> dict[str, np.ndarray]:
    train = np.load(fold_dir / "loss_tra.npy")
    epochs = int((~np.isnan(train).all(axis=1)).sum())  # Epochs completed before early stopping
    dice3d = np.load(fold_dir / "dice3d_val.npy")[:epochs, :, 1:]  # (epochs, patients, organs)
    organs = np.nanmean(dice3d, axis=1)
    return {"epoch": np.arange(1, epochs + 1),
            "train_loss": np.nanmean(train[:epochs], axis=1),
            "val_loss": np.nanmean(np.load(fold_dir / "loss_val.npy")[:epochs], axis=1),
            "organs": organs, "dice": organs.mean(axis=1)}


def load_swin(fold_dir: Path) -> dict[str, np.ndarray]:
    log = (fold_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    losses: dict[int, list[float]] = {}
    for epoch, loss in re.findall(r"Epoch: (\d+)/\d+, Iter: \d+/\d+ -- train_loss: ([\d.]+)", log):
        losses.setdefault(int(epoch), []).append(float(loss))
    val = [(int(e), float(d)) for e, d in re.findall(r"Epoch\[(\d+)\] Metrics -- .*?val_mean_dice: ([\d.]+)", log)]
    val_loss = [float(v) for v in re.findall(r"Epoch\[\d+\] Metrics -- .*?val_loss: ([\d.]+)", log)]
    return {"epoch": np.array(sorted(losses)), "train_loss": np.array([np.mean(losses[e]) for e in sorted(losses)]),
            "val_epoch": np.array([e for e, _ in val]), "dice": np.array([d for _, d in val]),
            "val_loss": np.array(val_loss) if len(val_loss) == len(val) else None}


def plot_25d(name: str, run_dir: Path, title: str, output: Path) -> dict[int, dict[str, np.ndarray]]:
    data = {fold: load_25d(run_dir / f"fold_{fold}") for fold in FOLDS}
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4), sharex=True, sharey="row", facecolor=SURFACE)
    for fold, d in data.items():
        top, bottom = axes[0, fold], axes[1, fold]
        style(top); style(bottom)
        # Neutral ink for the losses: in these figures colour means organ
        top.plot(d["epoch"], d["train_loss"], color=INK_MUTED, linewidth=2, label="Train loss")
        top.plot(d["epoch"], d["val_loss"], color=INK_MUTED, linewidth=2, linestyle="--", label="Validation loss")
        top.set_yscale("log")
        log_ticks(top)
        top.set_title(f"Fold {fold}", color=INK, fontsize=10)

        best = int(np.nanargmax(d["dice"]))
        for colour, organ, values in zip(ORGAN_COLOURS, ORGANS, d["organs"].T):
            bottom.plot(d["epoch"], values, color=colour, linewidth=1.5, label=organ)
        bottom.plot(d["epoch"], d["dice"], color=INK, linewidth=2.2, label="Mean of the organs")
        bottom.axvline(d["epoch"][best], color=INK_MUTED, linewidth=1, linestyle=":")
        bottom.annotate(f"best {d['dice'][best]:.3f}\nepoch {d['epoch'][best]}", (d["epoch"][best], 0.42),
                        xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK_MUTED)
        bottom.set_ylim(0.4, 1.0)
        bottom.set_xlabel("Epoch", color=INK_MUTED, fontsize=9)
    axes[0, 0].set_ylabel("Loss (log scale)", color=INK_MUTED, fontsize=9)
    axes[1, 0].set_ylabel("Validation 3D Dice", color=INK_MUTED, fontsize=9)

    handles = axes[0, 0].get_legend_handles_labels()[0] + axes[1, 0].get_legend_handles_labels()[0]
    labels = axes[0, 0].get_legend_handles_labels()[1] + axes[1, 0].get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{title} ({name})", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output / f"{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return data


def plot_swin(name: str, run_dir: Path, title: str, output: Path) -> None:
    has_val_loss = all(load_swin(run_dir / f"fold_{fold}")["val_loss"] is not None for fold in FOLDS)
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4), sharex=True, sharey="row", facecolor=SURFACE)
    for fold in FOLDS:
        d = load_swin(run_dir / f"fold_{fold}")
        top, bottom = axes[0, fold], axes[1, fold]
        style(top); style(bottom)
        top.plot(d["epoch"], d["train_loss"], color=INK_MUTED, linewidth=1.2, label="Train loss (mean per epoch)")
        if has_val_loss:
            top.plot(d["val_epoch"], d["val_loss"], color=INK_MUTED, linewidth=2, linestyle="--", marker="o",
                     markersize=3, label="Validation loss (every 10 epochs)")
        top.set_yscale("log")
        log_ticks(top)
        top.set_title(f"Fold {fold}", color=INK, fontsize=10)

        best = int(np.argmax(d["dice"]))
        bottom.plot(d["val_epoch"], d["dice"], color=INK, linewidth=2.2, marker="o", markersize=3,
                    label="Mean validation Dice (every 10 epochs)")
        bottom.axvline(d["val_epoch"][best], color=INK_MUTED, linewidth=1, linestyle=":")
        bottom.annotate(f"best {d['dice'][best]:.3f}\nepoch {d['val_epoch'][best]}", (d["val_epoch"][best], 0.62),
                        xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK_MUTED)
        bottom.set_ylim(0.6, 0.95)
        bottom.set_xlabel("Epoch", color=INK_MUTED, fontsize=9)
    axes[0, 0].set_ylabel("Loss (log scale)" if has_val_loss else "Train loss (log scale)", color=INK_MUTED, fontsize=9)
    axes[1, 0].set_ylabel("Validation Dice (1.5 mm space)", color=INK_MUTED, fontsize=9)
    handles = axes[0, 0].get_legend_handles_labels()[0] + axes[1, 0].get_legend_handles_labels()[0]
    labels = axes[0, 0].get_legend_handles_labels()[1] + axes[1, 0].get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{title} ({name})" + ("" if has_val_loss else ": this run did not log a validation loss"),
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output / f"{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


MIN_FOLDS = 3  # Stop a mean curve when fewer folds are still training: it would be the mean of one or two folds


def plot_overview(all_data: dict[str, dict[int, dict[str, np.ndarray]]], output: Path) -> None:
    labels = {"ce_dice_25d": "CE + Dice (up to 35 epochs)",
              "ce_dice_25d_aug": "CE + Dice, online augmentation (up to 50 epochs)",
              "weighted_ce_25d_e50": "Weighted CE (up to 50 epochs)"}
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style(ax)
    for colour, (name, data) in zip((BLUE, ORANGE, VIOLET), all_data.items()):
        longest = max(len(d["dice"]) for d in data.values())
        curves = np.full((len(data), longest), np.nan)
        for i, d in enumerate(data.values()):
            curves[i, :len(d["dice"])] = d["dice"]
        running = (~np.isnan(curves)).sum(axis=0)
        epochs = np.arange(1, longest + 1)
        all_folds = running == len(data)
        some_folds = running >= MIN_FOLDS
        mean = np.nanmean(curves, axis=0)
        ax.plot(epochs[all_folds], mean[all_folds], color=colour, linewidth=2, label=labels[name])
        if (some_folds & ~all_folds).any():  # Some folds stopped early: dotted while at least MIN_FOLDS remain
            tail = np.nonzero(some_folds)[0]
            tail = tail[tail >= np.nonzero(all_folds)[0][-1]]
            ax.plot(epochs[tail], mean[tail], color=colour, linewidth=1.5, linestyle=":")
    ax.set_ylim(0.6, 0.9)
    ax.set_xlabel("Epoch", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Validation 3D Dice, mean over folds", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    ax.set_title("2.5D runs: mean validation 3D Dice per epoch\n"
                 f"solid: all 5 folds training, dotted: {MIN_FOLDS}-4 folds left after early stopping",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(output / "overview_25d_val_dice.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("results/comparison/curves"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_data = {name: plot_25d(name, Path(run_dir), title, args.output) for name, (run_dir, title) in RUNS_25D.items()}
    for name, (run_dir, title) in SWIN_RUNS.items():
        if all((Path(run_dir) / f"fold_{fold}" / "log.txt").is_file() for fold in FOLDS):
            plot_swin(name, Path(run_dir), title, args.output)
        else:
            print(f"Skipping {name}: not all folds have results in {run_dir}")
    plot_overview(all_data, args.output)
    print(f"Saved figures to {args.output}: " + ", ".join(sorted(p.name for p in args.output.glob("*.png"))))


if __name__ == "__main__":
    main()
