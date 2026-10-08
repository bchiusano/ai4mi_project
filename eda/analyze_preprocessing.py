#!/usr/bin/env python3
"""Create compact EDA figures that motivate SegTHOR preprocessing choices."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Rectangle
from skimage.transform import resize


PLOT_PATIENTS = ("Patient_05", "Patient_19", "Patient_30", "Patient_40")
CROP_FIELDS_OF_VIEW_MM = (320, 352, 384, 416)
OUTPUT_SHAPE = (384, 384)
TARGET_IN_PLANE_SPACING_MM = (1.0, 1.0)
HU_WINDOW = (-1000.0, 1000.0)


def centre_crop(array: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    crop_x, crop_y = (size, size) if isinstance(size, int) else size
    x0 = (array.shape[0] - crop_x) // 2
    y0 = (array.shape[1] - crop_y) // 2
    return array[x0 : x0 + crop_x, y0 : y0 + crop_y, ...]


def resize_image(array: np.ndarray) -> np.ndarray:
    return resize(
        array,
        OUTPUT_SHAPE,
        order=1,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    )


def resize_label(array: np.ndarray) -> np.ndarray:
    return resize(
        array,
        OUTPUT_SHAPE,
        order=0,
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)


def fixed_hu_normalize(array: np.ndarray) -> np.ndarray:
    low, high = HU_WINDOW
    return (np.clip(array.astype(np.float32), low, high) - low) / (high - low)


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coordinates = np.where(mask > 0)
    return (
        int(coordinates[0].min()),
        int(coordinates[0].max()),
        int(coordinates[1].min()),
        int(coordinates[1].max()),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted(path for path in (args.full_root / "train").glob("Patient_*") if path.is_dir())
    if len(patient_dirs) != 40:
        raise RuntimeError(f"Expected 40 official patients, found {len(patient_dirs)}")

    cohort_rows: list[dict[str, object]] = []
    plot_examples: dict[str, dict[str, np.ndarray | int]] = {}

    for index, patient_dir in enumerate(patient_dirs, start=1):
        patient = patient_dir.name
        print(f"[{index:02d}/{len(patient_dirs):02d}] {patient}", flush=True)
        ct_image = nib.load(str(patient_dir / f"{patient}.nii.gz"))
        gt_image = nib.load(str(patient_dir / "GT.nii.gz"))
        ct = np.asanyarray(ct_image.dataobj)
        gt = np.asanyarray(gt_image.dataobj).astype(np.uint8)
        spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
        bbox = foreground_bbox(gt)
        foreground = gt > 0
        total_foreground = int(foreground.sum())
        sample = ct[::8, ::8, ::4]
        percentiles = np.percentile(sample, [0.5, 1, 50, 99, 99.5])

        row: dict[str, object] = {
            "patient": patient,
            "slices": ct.shape[2],
            "spacing_x_mm": spacing[0],
            "spacing_y_mm": spacing[1],
            "spacing_z_mm": spacing[2],
            "ct_min_hu": int(ct.min()),
            "ct_max_hu": int(ct.max()),
            "p00_5_hu": float(percentiles[0]),
            "p01_hu": float(percentiles[1]),
            "p50_hu": float(percentiles[2]),
            "p99_hu": float(percentiles[3]),
            "p99_5_hu": float(percentiles[4]),
            "zero_hu_current_uint8": 255.0 * (0.0 - float(ct.min())) / (float(ct.max()) - float(ct.min())),
            "bbox_x_min": bbox[0],
            "bbox_x_max": bbox[1],
            "bbox_y_min": bbox[2],
            "bbox_y_max": bbox[3],
        }
        for field_of_view_mm in CROP_FIELDS_OF_VIEW_MM:
            crop_shape = tuple(
                int(round(field_of_view_mm / spacing[axis])) for axis in (0, 1)
            )
            kept = int(np.count_nonzero(centre_crop(foreground, crop_shape)))
            row[f"fov_{field_of_view_mm}_retained_percent"] = 100.0 * kept / total_foreground
        cohort_rows.append(row)

        if patient in PLOT_PATIENTS:
            areas = np.count_nonzero(foreground, axis=(0, 1))
            z_index = int(np.argmax(areas))
            raw_slice = ct[:, :, z_index]
            gt_slice = gt[:, :, z_index]
            minmax_slice = resize_image(
                (raw_slice.astype(np.float32) - float(ct.min()))
                / (float(ct.max()) - float(ct.min()))
            )
            direct_gt = resize_label(gt_slice)
            physical_crop_shape = tuple(
                int(round(OUTPUT_SHAPE[axis] * TARGET_IN_PLANE_SPACING_MM[axis] / spacing[axis]))
                for axis in (0, 1)
            )
            cropped_slice = centre_crop(raw_slice, physical_crop_shape)
            cropped_gt = centre_crop(gt_slice, physical_crop_shape)
            windowed = resize_image(fixed_hu_normalize(cropped_slice))
            resized_gt = resize_label(cropped_gt)
            plot_examples[patient] = {
                "z": z_index,
                "current": minmax_slice,
                "current_gt": direct_gt,
                "windowed": windowed,
                "cropped_gt": resized_gt,
            }

        # The NIfTI volumes are large. Release each patient's arrays and image
        # proxies before loading the next patient so the EDA also works on
        # laptops with limited RAM.
        del ct, gt, ct_image, gt_image, foreground, sample
        gc.collect()

    write_csv(args.output / "full_cohort_preprocessing_summary.csv", cohort_rows)
    make_before_after_plot(plot_examples, args.output)
    make_intensity_plot(cohort_rows, args.output)
    make_crop_plot(cohort_rows, args.output)
    make_spacing_plot(cohort_rows, args.output)

    summary = {
        "patients": len(cohort_rows),
        "archive_sha256": "53e99dfb45234a9ed4ac573db767cb5e3179da209c6a5bf1ddc999bce03f4e51",
        "spacing_x_range_mm": [min(row["spacing_x_mm"] for row in cohort_rows), max(row["spacing_x_mm"] for row in cohort_rows)],
        "spacing_z_values_mm": sorted({row["spacing_z_mm"] for row in cohort_rows}),
        "fov_384_min_retained_percent": min(row["fov_384_retained_percent"] for row in cohort_rows),
        "current_zero_hu_uint8_range": [min(row["zero_hu_current_uint8"] for row in cohort_rows), max(row["zero_hu_current_uint8"] for row in cohort_rows)],
        "fixed_zero_hu_uint8": 127.5,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def add_contours(axis: plt.Axes, label: np.ndarray) -> None:
    axis.contour(np.rot90(label), levels=[0.5, 1.5, 2.5, 3.5], colors=["#00e5ff", "#ffcc00", "#ff4d8d", "#39ff88"], linewidths=0.7)


def make_before_after_plot(examples: dict[str, dict[str, np.ndarray | int]], output: Path) -> None:
    figure, axes = plt.subplots(len(PLOT_PATIENTS), 2, figsize=(8, 12), constrained_layout=True)
    column_titles = (
        "Current: full slice + per-scan min-max",
        "Selected: 384 mm crop + fixed HU window",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, weight="bold")
    for row, patient in enumerate(PLOT_PATIENTS):
        example = examples[patient]
        images = (example["current"], example["windowed"])
        labels = (example["current_gt"], example["cropped_gt"])
        for column, (image, label) in enumerate(zip(images, labels)):
            axes[row, column].imshow(np.rot90(image), cmap="gray", vmin=0, vmax=1)
            add_contours(axes[row, column], label)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"{patient}\nslice {example['z']}", fontsize=10)
    figure.suptitle("Representative SegTHOR slices before and after selected preprocessing\nColored contours show ground-truth organ boundaries", fontsize=14, weight="bold")
    figure.savefig(output / "preprocessing_before_after.png", dpi=180, facecolor="white")
    plt.close(figure)


def make_intensity_plot(rows: list[dict[str, object]], output: Path) -> None:
    patients = np.arange(1, len(rows) + 1)
    maxima = np.array([row["ct_max_hu"] for row in rows], dtype=float)
    zero_current = np.array([row["zero_hu_current_uint8"] for row in rows], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].scatter(patients, maxima, color="#3569b7", s=28)
    axes[0].axhline(HU_WINDOW[1], color="#c83e4d", linestyle="--", label="Fixed upper window: 1000 HU")
    axes[0].set(xlabel="Patient", ylabel="Maximum CT value (HU)", title="Extreme intensities differ strongly")
    axes[0].legend(frameon=False)
    axes[1].scatter(patients, zero_current, color="#e08b26", s=28, label="Current per-scan min-max")
    axes[1].axhline(127.5, color="#2f8f5b", linestyle="--", label="Fixed HU window")
    axes[1].set(xlabel="Patient", ylabel="Encoded value for 0 HU (0–255)", title="Per-scan min-max changes tissue meaning")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output / "intensity_normalization_motivation.png", dpi=180, facecolor="white")
    plt.close(figure)


def make_crop_plot(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True)
    axes[0].set_xlim(0, 512)
    axes[0].set_ylim(512, 0)
    axes[0].set_aspect("equal")
    for row in rows:
        width = row["bbox_y_max"] - row["bbox_y_min"] + 1
        height = row["bbox_x_max"] - row["bbox_x_min"] + 1
        axes[0].add_patch(Rectangle((row["bbox_y_min"], row["bbox_x_min"]), width, height, fill=False, edgecolor="#4472c4", linewidth=0.55, alpha=0.35))
    reference_spacing = float(np.median([row["spacing_x_mm"] for row in rows]))
    reference_crop = int(round(384 / reference_spacing))
    crop_start = (512 - reference_crop) // 2
    axes[0].add_patch(Rectangle((crop_start, crop_start), reference_crop, reference_crop, fill=False, edgecolor="#d62728", linewidth=2.2, label="384 mm crop at median spacing"))
    axes[0].set(title="All 40 foreground bounding boxes", xlabel="Image column", ylabel="Image row")
    axes[0].legend(frameon=False)

    for field_of_view_mm, color in zip(CROP_FIELDS_OF_VIEW_MM, ["#c44e52", "#dd8452", "#55a868", "#4c72b0"]):
        retained = [row[f"fov_{field_of_view_mm}_retained_percent"] for row in rows]
        axes[1].scatter([field_of_view_mm] * len(retained), retained, color=color, alpha=0.6, s=22)
        axes[1].plot(field_of_view_mm, min(retained), marker="D", color="black", markersize=5)
    axes[1].set(xlabel="Centred physical field of view (mm)", ylabel="Foreground voxels retained (%)", title="Crop safety across patients", xticks=CROP_FIELDS_OF_VIEW_MM, ylim=(97, 100.1))
    axes[1].grid(alpha=0.2)
    figure.savefig(output / "crop_motivation_and_safety.png", dpi=180, facecolor="white")
    plt.close(figure)


def make_spacing_plot(rows: list[dict[str, object]], output: Path) -> None:
    patients = np.arange(1, len(rows) + 1)
    in_plane = np.array([row["spacing_x_mm"] for row in rows])
    through_plane = np.array([row["spacing_z_mm"] for row in rows])
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.scatter(patients, in_plane, label="In-plane x/y spacing", color="#3569b7", s=30)
    axis.scatter(patients, through_plane, label="Slice spacing (z)", color="#c83e4d", marker="x", s=38)
    axis.axhline(1.0, color="#2f8f5b", linestyle="--", linewidth=1.5, label="Chosen in-plane target: 1.0 mm")
    axis.set(xlabel="Patient", ylabel="Voxel spacing (mm)", title="Voxel spacing is not identical across patients")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output / "voxel_spacing_by_patient.png", dpi=180, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=Path("data/segthor_train_full"))
    parser.add_argument("--output", type=Path, default=Path("eda/preprocessing_figures"))
    return parser.parse_args()


if __name__ == "__main__":
    analyse(parse_args())
