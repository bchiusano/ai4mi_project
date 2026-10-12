#!/usr/bin/env python3
"""Slice unlabelled CT volumes (e.g. the SegTHOR test set) like the training data, for the 2D/2.5D models.

Applies the crop, HU window and resize of preprocessing/slice_segthor.py, with the settings read from the
preprocessing.json of the training dataset so the two cannot drift apart, and records each patient's crop so
predictions can be mapped back to the original CT (inference/ensemble_predictions.py).

Usage:
    python -m inference.prepare_test_slices --ct-dir <dir with Patient_XX.nii.gz> \
        --training-preprocessing-json data/SEGTHOR_FULL_PREPROCESSED/preprocessing.json \
        --dest data/SEGTHOR_TEST_PREPROCESSED
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.io import imsave

from preprocessing.slice_segthor import _resize_image, centre_crop, crop_shape_for_spacing, sanity_ct, window_ct


def slice_ct(ct_path: Path, dest: Path, settings: dict) -> dict:
    patient_id = re.match(r"Patient_\d+", ct_path.name).group(0)
    image = nib.load(str(ct_path))
    ct = np.asanyarray(image.dataobj)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    sanity_ct(ct, spacing)
    orientation = "".join(nib.aff2axcodes(image.affine))
    if orientation != "LPS":
        raise ValueError(f"{patient_id}: expected LPS orientation like the training scans, found {orientation}")

    shape = tuple(settings["output_shape"])
    crop_shape = crop_shape_for_spacing(ct.shape, spacing, shape, tuple(settings["requested_in_plane_spacing_mm"]))
    cropped, crop_bounds = centre_crop(ct, crop_shape)
    processed = window_ct(cropped, *settings["intensity_window_hu"])

    image_dir = dest / "img"
    for z_index in range(processed.shape[2]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(image_dir / f"{patient_id}_{z_index:04d}.png"), _resize_image(processed[:, :, z_index], shape))

    return {"patient": patient_id, "ct_path": str(ct_path.resolve()), "original_shape": list(ct.shape),
            "original_spacing_mm": list(spacing), "orientation": orientation, "crop_bounds": list(crop_bounds),
            "crop_shape": list(crop_shape), "output_shape": [*shape, int(processed.shape[2])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ct-dir", type=Path, required=True, help="Folder with Patient_XX.nii.gz CT volumes.")
    parser.add_argument("--training-preprocessing-json", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    settings = json.loads(args.training_preprocessing_json.read_text(encoding="utf-8"))
    ct_paths = sorted(args.ct_dir.glob("Patient_*.nii*"))
    if not ct_paths:
        raise RuntimeError(f"No Patient_*.nii* volumes in {args.ct_dir}")
    if (args.dest / "img").exists() and any((args.dest / "img").iterdir()):
        raise RuntimeError(f"Destination already has slices: {args.dest / 'img'}")
    (args.dest / "img").mkdir(parents=True, exist_ok=True)

    patients = []
    for path in ct_paths:
        patients.append(slice_ct(path, args.dest, settings))
        print(f"{patients[-1]['patient']}: {patients[-1]['output_shape'][2]} slices")

    metadata = {key: settings[key] for key in ("output_shape", "requested_in_plane_spacing_mm",
                                               "requested_field_of_view_mm", "intensity_window_hu")}
    metadata |= {"source_dir": str(args.ct_dir.resolve()), "settings_from": str(args.training_preprocessing_json),
                 "patients": patients}
    (args.dest / "preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(patients)} patients to {args.dest}")


if __name__ == "__main__":
    main()
