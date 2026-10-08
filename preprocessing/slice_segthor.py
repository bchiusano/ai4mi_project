#!/usr/bin/env python3

"""Convert SegTHOR NIfTI volumes into deterministic 2-D training slices.

The preprocessing is deliberately light and reproducible:

* clip CT values to a fixed Hounsfield-unit window and map it to uint8;
* crop the same physical in-plane field of view from every scan;
* resample images and labels to a common in-plane spacing;
* define one fixed patient-level train/validation/test split; and
* save the split, crop, intensity-window, and effective-spacing metadata.

The crop never depends on the ground-truth mask, so the same transform can be
used for unseen scans. For labelled data, preprocessing fails if the configured
crop would discard any foreground voxel.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Sequence

import nibabel as nib
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


SEGTHOR_LABELS = {0, 1, 2, 3, 4}
LABEL_PNG_SCALE = 63


def window_ct(
    image: np.ndarray,
    lower_hu: float = -1000.0,
    upper_hu: float = 1000.0,
) -> np.ndarray:
    """Clip a CT to a fixed HU window and map it to ``uint8`` [0, 255]."""

    if not lower_hu < upper_hu:
        raise ValueError("The lower HU bound must be smaller than the upper bound")

    clipped = np.clip(image.astype(np.float32), lower_hu, upper_hu)
    scaled = (clipped - lower_hu) * (255.0 / (upper_hu - lower_hu))
    return np.rint(scaled).astype(np.uint8)


def centre_crop_bounds(
    input_shape: Sequence[int], crop_shape: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return ``x0, x1, y0, y1`` for a centred in-plane crop."""

    if len(input_shape) < 2 or len(crop_shape) != 2:
        raise ValueError("Expected an input with at least two axes and a 2-D crop shape")

    input_x, input_y = (int(input_shape[0]), int(input_shape[1]))
    crop_x, crop_y = (int(crop_shape[0]), int(crop_shape[1]))
    if crop_x <= 0 or crop_y <= 0:
        raise ValueError("Crop dimensions must be positive")
    if crop_x > input_x or crop_y > input_y:
        raise ValueError(
            f"Crop shape {(crop_x, crop_y)} exceeds input shape {(input_x, input_y)}"
        )

    x0 = (input_x - crop_x) // 2
    y0 = (input_y - crop_y) // 2
    return x0, x0 + crop_x, y0, y0 + crop_y


def centre_crop(
    volume: np.ndarray, crop_shape: Sequence[int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bounds = centre_crop_bounds(volume.shape, crop_shape)
    x0, x1, y0, y1 = bounds
    return volume[x0:x1, y0:y1, ...], bounds


def crop_shape_for_spacing(
    input_shape: Sequence[int],
    input_spacing: Sequence[float],
    output_shape: Sequence[int],
    target_spacing: Sequence[float],
) -> tuple[int, int]:
    """Convert a requested physical output field of view to source pixels."""

    if len(input_spacing) < 2 or len(output_shape) != 2 or len(target_spacing) != 2:
        raise ValueError("Expected 2-D output shape/spacing and at least 2-D input spacing")
    if any(float(value) <= 0 for value in (*input_spacing[:2], *target_spacing)):
        raise ValueError("Voxel spacings must be positive")

    crop_shape = tuple(
        int(round(int(output_pixels) * float(output_mm) / float(input_mm)))
        for output_pixels, output_mm, input_mm in zip(
            output_shape, target_spacing, input_spacing[:2]
        )
    )
    centre_crop_bounds(input_shape, crop_shape)
    return crop_shape


def sanity_ct(ct: np.ndarray, spacing: Sequence[float]) -> None:
    if not np.issubdtype(ct.dtype, np.signedinteger):
        raise TypeError(f"Expected signed-integer CT values, found {ct.dtype}")
    if ct.ndim != 3:
        raise ValueError(f"Expected a 3-D CT, found shape {ct.shape}")
    if not np.all(np.isfinite(ct)):
        raise ValueError("CT contains non-finite values")
    if len(spacing) < 3 or any(float(value) <= 0 for value in spacing[:3]):
        raise ValueError(f"Invalid voxel spacing: {spacing}")


def sanity_gt(gt: np.ndarray, ct: np.ndarray) -> None:
    if gt.shape != ct.shape:
        raise ValueError(f"GT shape {gt.shape} differs from CT shape {ct.shape}")
    if not np.issubdtype(gt.dtype, np.integer):
        raise TypeError(f"Expected integer labels, found {gt.dtype}")
    labels = set(np.unique(gt).astype(int))
    if not labels.issubset(SEGTHOR_LABELS):
        raise ValueError(f"Unexpected SegTHOR labels: {sorted(labels)}")


def _resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = resize(
        image,
        shape,
        order=1,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    )
    return np.rint(resized).astype(np.uint8)


def _resize_label(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(
        label,
        shape,
        order=0,
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)


def slice_patient(
    id_: str,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    target_spacing: tuple[float, float],
    intensity_window: tuple[float, float],
) -> dict[str, object]:
    id_path = source_path / "train" / id_
    ct_path = id_path / f"{id_}.nii.gz"

    ct_image = nib.load(str(ct_path))
    ct = np.asanyarray(ct_image.dataobj)
    spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
    sanity_ct(ct, spacing)
    orientation = "".join(nib.aff2axcodes(ct_image.affine))
    if orientation != "LPS":
        raise ValueError(f"{id_}: expected LPS orientation, found {orientation}")

    gt_image = nib.load(str(id_path / "GT.nii.gz"))
    gt = np.asanyarray(gt_image.dataobj)
    sanity_gt(gt, ct)
    if not np.allclose(ct_image.affine, gt_image.affine, atol=1e-5):
        raise ValueError(f"{id_}: CT and GT affine matrices differ")

    crop_shape = crop_shape_for_spacing(ct.shape, spacing, shape, target_spacing)
    cropped_ct, crop_bounds = centre_crop(ct, crop_shape)
    cropped_gt, gt_bounds = centre_crop(gt, crop_shape)
    if gt_bounds != crop_bounds:
        raise AssertionError("CT and GT crop bounds differ")
    original_foreground = int(np.count_nonzero(gt))
    cropped_foreground = int(np.count_nonzero(cropped_gt))
    if cropped_foreground != original_foreground:
        raise ValueError(
            f"{id_}: crop would discard "
            f"{original_foreground - cropped_foreground} foreground voxels"
        )

    lower_hu, upper_hu = intensity_window
    processed_ct = window_ct(cropped_ct, lower_hu, upper_hu)
    z_slices = int(processed_ct.shape[2])

    image_dir = dest_path / "img"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir = dest_path / "gt"
    label_dir.mkdir(parents=True, exist_ok=True)

    for z_index in range(z_slices):
        image_slice = _resize_image(processed_ct[:, :, z_index], shape)
        filename = f"{id_}_{z_index:04d}.png"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(image_dir / filename), image_slice)

        label_slice = _resize_label(cropped_gt[:, :, z_index], shape)
        if not set(np.unique(label_slice).astype(int)).issubset(SEGTHOR_LABELS):
            raise AssertionError(f"{id_}: label interpolation introduced invalid classes")
        encoded_label = (label_slice * LABEL_PNG_SCALE).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(label_dir / filename), encoded_label)

    crop_x, crop_y = crop_shape
    output_x, output_y = shape
    effective_spacing = (
        spacing[0] * crop_x / output_x,
        spacing[1] * crop_y / output_y,
        spacing[2],
    )
    return {
        "patient": id_,
        "original_shape": [int(value) for value in ct.shape],
        "original_spacing_mm": list(spacing),
        "orientation": orientation,
        "crop_bounds": list(crop_bounds),
        "crop_shape": [int(crop_x), int(crop_y)],
        "requested_field_of_view_mm": [
            float(shape[0] * target_spacing[0]),
            float(shape[1] * target_spacing[1]),
        ],
        "requested_in_plane_spacing_mm": list(target_spacing),
        "output_shape": [int(output_x), int(output_y), z_slices],
        "effective_spacing_mm": [float(value) for value in effective_spacing],
        "ct_min_hu": int(ct.min()),
        "ct_max_hu": int(ct.max()),
    }


def get_patient_split(
    source_path: Path,
    validation_count: int,
    test_count: int,
    seed: int,
    test_pool_start: int | None = 21,
) -> dict[str, object]:
    """Create one deterministic, stratified train/validation/test split."""

    ids = sorted(path.name for path in (source_path / "train").glob("Patient_*") if path.is_dir())
    if not ids:
        raise RuntimeError(f"No Patient_* directories found in {source_path / 'train'}")
    if test_count <= 0 or test_count >= len(ids):
        raise ValueError(f"test_count must be between 1 and {len(ids) - 1}")
    development_count = len(ids) - test_count
    if validation_count <= 0 or validation_count >= development_count:
        raise ValueError(
            f"validation_count must be between 1 and {development_count - 1}"
        )

    if test_pool_start is None:
        test_candidates = ids.copy()
    else:
        test_candidates = [
            patient
            for patient in ids
            if int(patient.rsplit("_", maxsplit=1)[1]) >= test_pool_start
        ]
    if len(test_candidates) < test_count:
        raise ValueError(
            f"Only {len(test_candidates)} patients are eligible for a "
            f"{test_count}-patient test set"
        )

    random_generator = random.Random(seed)
    random_generator.shuffle(test_candidates)
    test_ids = test_candidates[:test_count]
    test_set = set(test_ids)
    if test_pool_start is None:
        development_strata = [[patient for patient in ids if patient not in test_set]]
    else:
        preliminary_ids = [
            patient
            for patient in ids
            if int(patient.rsplit("_", maxsplit=1)[1]) < test_pool_start
        ]
        new_development_ids = [
            patient for patient in test_candidates if patient not in test_set
        ]
        development_strata = [preliminary_ids, new_development_ids]
    for stratum in development_strata:
        random_generator.shuffle(stratum)

    exact_allocations = [
        validation_count * len(stratum) / development_count
        for stratum in development_strata
    ]
    validation_allocations = [int(value) for value in exact_allocations]
    remaining = validation_count - sum(validation_allocations)
    allocation_order = sorted(
        range(len(development_strata)),
        key=lambda index: exact_allocations[index] - validation_allocations[index],
        reverse=True,
    )
    for index in allocation_order[:remaining]:
        validation_allocations[index] += 1

    validation_ids = [
        patient
        for stratum, count in zip(development_strata, validation_allocations)
        for patient in stratum[:count]
    ]
    training_ids = [
        patient
        for stratum, count in zip(development_strata, validation_allocations)
        for patient in stratum[count:]
    ]
    random_generator.shuffle(training_ids)
    random_generator.shuffle(validation_ids)

    print(f"Found {len(ids)} labelled patients")
    print(f"Training patients ({len(training_ids)}): {training_ids}")
    print(f"Validation patients ({len(validation_ids)}): {validation_ids}")
    print(f"Fixed test patients ({len(test_ids)}): {test_ids}")
    return {
        "seed": seed,
        "training_patient_count": len(training_ids),
        "validation_patient_count": validation_count,
        "test_patient_count": test_count,
        "test_pool_start": test_pool_start,
        "train": training_ids,
        "val": validation_ids,
        "test": test_ids,
    }


def get_splits(
    source_path: Path,
    validation_count: int = 6,
    test_count: int = 6,
    seed: int = 0,
    test_pool_start: int | None = 21,
) -> tuple[list[str], list[str], list[str]]:
    """Return the fixed train, validation and test patient IDs."""

    manifest = get_patient_split(
        source_path, validation_count, test_count, seed, test_pool_start
    )
    return list(manifest["train"]), list(manifest["val"]), list(manifest["test"])


def _process_patients(
    patient_ids: list[str],
    *,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    target_spacing: tuple[float, float],
    intensity_window: tuple[float, float],
    process_count: int,
) -> list[dict[str, object]]:
    if not patient_ids:
        return []

    print(f"Slicing {len(patient_ids)} patients to {dest_path}")
    process_patient: Callable = partial(
        slice_patient,
        dest_path=dest_path,
        source_path=source_path,
        shape=shape,
        target_spacing=target_spacing,
        intensity_window=intensity_window,
    )

    iterator = tqdm_(patient_ids)
    if process_count == 1:
        return list(map(process_patient, iterator))
    if process_count == -1:
        with Pool() as pool:
            return pool.map(process_patient, iterator)
    with Pool(process_count) as pool:
        return pool.map(process_patient, iterator)


def main(args: argparse.Namespace) -> None:
    source_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")
    if dest_path.exists():
        raise FileExistsError(f"Destination already exists: {dest_path}")

    shape = tuple(args.shape)
    target_spacing = tuple(args.target_spacing)
    intensity_window = tuple(args.window)
    split_manifest = get_patient_split(
        source_path,
        args.validation_count,
        args.test_count,
        args.split_seed,
        args.test_pool_start,
    )
    all_ids = (
        list(split_manifest["train"])
        + list(split_manifest["val"])
        + list(split_manifest["test"])
    )

    dest_path.mkdir(parents=True)
    all_metadata = _process_patients(
        all_ids,
        dest_path=dest_path,
        source_path=source_path,
        shape=shape,
        target_spacing=target_spacing,
        intensity_window=intensity_window,
        process_count=args.process,
    )

    spacing = {
        str(row["patient"]): tuple(row["effective_spacing_mm"])
        for row in all_metadata
    }
    with (dest_path / "spacing.pkl").open("wb") as spacing_file:
        pickle.dump(spacing, spacing_file, pickle.HIGHEST_PROTOCOL)

    with (dest_path / "split.json").open("w", encoding="utf-8") as split_file:
        json.dump(split_manifest, split_file, indent=2)

    with (dest_path / "preprocessing.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "source_dir": str(source_path),
                "output_shape": list(shape),
                "requested_in_plane_spacing_mm": list(target_spacing),
                "requested_field_of_view_mm": [
                    float(shape[0] * target_spacing[0]),
                    float(shape[1] * target_spacing[1]),
                ],
                "intensity_window_hu": list(intensity_window),
                "image_interpolation_order": 1,
                "label_interpolation_order": 0,
                "patients": all_metadata,
            },
            config_file,
            indent=2,
        )

    print(f"Saved processed dataset to: {dest_path}")
    print(f"Saved split metadata to: {dest_path / 'split.json'}")
    print(f"Saved preprocessing metadata to: {dest_path / 'preprocessing.json'}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--dest_dir", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=2, default=[384, 384])
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("X_MM", "Y_MM"),
        help="Requested output in-plane spacing in millimetres (default: 1.0 1.0).",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=[-1000.0, 1000.0],
        metavar=("LOWER_HU", "UPPER_HU"),
        help="Fixed CT intensity window before uint8 conversion.",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=6,
        help="Number of patients used for validation (default: 6).",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=6,
        help="Number of patients held out for final testing (default: 6).",
    )
    parser.add_argument(
        "--test-pool-start",
        type=int,
        default=21,
        help=(
            "Only patient numbers at or above this value are eligible for the test set "
            "(default: 21, because Patients 1-20 were used in preliminary work)."
        ),
    )
    parser.add_argument(
        "--split-seed",
        "--seed",
        dest="split_seed",
        type=int,
        default=0,
        help=(
            "Seed used only to create the fixed patient split (default: 0). "
            "Do not vary this with training run seeds."
        ),
    )
    parser.add_argument(
        "--process",
        "-p",
        type=int,
        default=1,
        help="Worker processes; use -1 for all available CPUs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
