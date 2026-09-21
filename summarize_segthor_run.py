#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


ORGAN_NAMES = ("background", "esophagus", "heart", "trachea", "aorta")


def completed_epochs(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).any(axis=tuple(range(1, values.ndim)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed SegTHOR training run")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    dice3d = np.load(args.run_dir / "dice3d_val.npy")
    loss_train = np.load(args.run_dir / "loss_tra.npy")
    loss_val = np.load(args.run_dir / "loss_val.npy")

    valid = completed_epochs(dice3d)
    epochs = np.flatnonzero(valid)
    if epochs.size == 0:
        raise RuntimeError(f"No completed epochs in {args.run_dir}")

    foreground_by_epoch = np.nanmean(dice3d[:, :, 1:], axis=(1, 2))
    best_epoch = int(epochs[np.nanargmax(foreground_by_epoch[valid])])
    organ_scores = np.nanmean(dice3d[best_epoch, :, :], axis=0)

    print(f"Run directory: {args.run_dir}")
    print(f"Completed epochs: {len(epochs)}")
    print(f"Best 3D-Dice epoch: {best_epoch}")
    print(f"Mean foreground 3D Dice: {foreground_by_epoch[best_epoch]:.6f}")
    print(f"Training loss at best epoch: {np.nanmean(loss_train[best_epoch]):.6f}")
    print(f"Validation loss at best epoch: {np.nanmean(loss_val[best_epoch]):.6f}")
    for class_index, organ in enumerate(ORGAN_NAMES[1:], start=1):
        print(f"{organ.capitalize()} 3D Dice: {organ_scores[class_index]:.6f}")


if __name__ == "__main__":
    main()
