#!/usr/bin/env python3
"""Validate a sliced SegTHOR dataset and its preprocessing metadata."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ENCODED_LABELS = {0, 63, 126, 189, 252}
FOREGROUND_LABELS = {63, 126, 189, 252}
PATIENT_PATTERN = re.compile(r"^(Patient_\d\d)_\d{4}$")


def patient_from_stem(stem: str) -> str:
    match = PATIENT_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unexpected slice filename: {stem}")
    return match.group(1)


def validate(processed_dir: Path) -> dict[str, object]:
    split_path = processed_dir / "split.json"
    metadata_path = processed_dir / "preprocessing.json"
    spacing_path = processed_dir / "spacing.pkl"
    for path in (split_path, metadata_path, spacing_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing preprocessing output: {path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with spacing_path.open("rb") as spacing_file:
        spacing = pickle.load(spacing_file)

    train_ids = set(split["train"])
    validation_ids = set(split["val"])
    test_ids = set(split["test"])
    if len(train_ids) != split["training_patient_count"]:
        raise AssertionError("The saved training cohort size is incorrect")
    if len(validation_ids) != split["validation_patient_count"]:
        raise AssertionError("The saved validation cohort size is incorrect")
    if len(test_ids) != split["test_patient_count"]:
        raise AssertionError("The saved test cohort size differs from test_patient_count")
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Train, validation, and test patients overlap")

    patient_metadata = {row["patient"]: row for row in metadata["patients"]}
    expected_ids = train_ids | validation_ids | test_ids
    if set(patient_metadata) != expected_ids or set(spacing) != expected_ids:
        raise AssertionError("Patient IDs differ between split, spacing, and metadata files")

    total_images = 0
    total_labels = 0
    output_shape = tuple(metadata["output_shape"])
    requested_spacing = np.asarray(metadata["requested_in_plane_spacing_mm"], dtype=float)
    seen_ids: set[str] = set()

    image_files = sorted((processed_dir / "img").glob("*.png"))
    label_files = sorted((processed_dir / "gt").glob("*.png"))
    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}
    if image_stems != label_stems:
        raise AssertionError("Image/label filenames do not match")

    patient_slice_counts: dict[str, int] = defaultdict(int)
    patient_label_values: dict[str, set[int]] = defaultdict(set)
    for image_path in image_files:
        patient = patient_from_stem(image_path.stem)
        if patient not in expected_ids:
            raise AssertionError(f"Unexpected patient in processed images: {patient}")
        image = np.asarray(Image.open(image_path))
        if image.shape != output_shape:
            raise AssertionError(f"{image_path}: found shape {image.shape}, expected {output_shape}")
        if image.dtype != np.uint8:
            raise AssertionError(f"{image_path}: expected uint8, found {image.dtype}")
        patient_slice_counts[patient] += 1
        seen_ids.add(patient)

    for label_path in label_files:
        patient = patient_from_stem(label_path.stem)
        label = np.asarray(Image.open(label_path))
        if label.shape != output_shape:
            raise AssertionError(f"{label_path}: found shape {label.shape}, expected {output_shape}")
        values = set(np.unique(label).astype(int))
        if not values.issubset(ENCODED_LABELS):
            raise AssertionError(f"{label_path}: unexpected encoded labels {sorted(values)}")
        patient_label_values[patient].update(values)

    for patient in expected_ids:
        expected_slices = int(patient_metadata[patient]["output_shape"][2])
        if patient_slice_counts[patient] != expected_slices:
            raise AssertionError(
                f"{patient}: found {patient_slice_counts[patient]} slices, expected {expected_slices}"
            )
        if not FOREGROUND_LABELS.issubset(patient_label_values[patient]):
            missing = sorted(FOREGROUND_LABELS - patient_label_values[patient])
            raise AssertionError(f"{patient}: processed labels are missing {missing}")

    total_images = len(image_files)
    total_labels = len(label_files)

    if seen_ids != expected_ids:
        raise AssertionError(f"Missing processed patients: {sorted(expected_ids - seen_ids)}")

    effective_xy = np.array([spacing[patient][:2] for patient in sorted(spacing)], dtype=float)
    max_spacing_error = float(np.max(np.abs(effective_xy - requested_spacing)))
    if max_spacing_error > 0.01:
        raise AssertionError(f"Maximum in-plane spacing error is too large: {max_spacing_error}")

    return {
        "patients": len(expected_ids),
        "split_seed": split["seed"],
        "training_patients": len(train_ids),
        "validation_patients": len(validation_ids),
        "test_patients": len(test_ids),
        "image_slices": total_images,
        "label_slices": total_labels,
        "output_shape": list(output_shape),
        "requested_in_plane_spacing_mm": requested_spacing.tolist(),
        "effective_spacing_x_mm_range": [
            float(effective_xy[:, 0].min()),
            float(effective_xy[:, 0].max()),
        ],
        "max_requested_spacing_error_mm": max_spacing_error,
        "validation_status": "passed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    result = validate(parse_args().processed_dir)
    print(json.dumps(result, indent=2))
