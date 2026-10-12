#!/usr/bin/env python3
"""Convert 3D NIfTI predictions to the 2D PNG slices of the preprocessed dataset.

3D models (e.g. the MONAI SwinUNETR bundle) predict in the original CT space. Applying the exact crop and
label resize of preprocessing/slice_segthor.py, with each patient's crop from preprocessing.json, puts their
predictions on the same 384x384 slice grid as the 2D/2.5D models, so evaluation/evaluate_saved_predictions.py
scores both against the same ground-truth slices.

Usage:
    python -m evaluation.nifti_predictions_to_slices \
        --pred-dir results/segthor_swin_unetr/cv/fold_0/eval \
        --preprocessing-json data/SEGTHOR_FULL_PREPROCESSED/preprocessing.json \
        --output-dir results/segthor_swin_unetr/cv/fold_0/eval_slices
    python -m evaluation.evaluate_saved_predictions \
        --pred-dir results/segthor_swin_unetr/cv/fold_0/eval_slices \
        --gt-dir data/SEGTHOR_FULL_PREPROCESSED_split_cv/fold_0/val/gt
"""

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from preprocessing.slice_segthor import LABEL_PNG_SCALE, SEGTHOR_LABELS, _resize_label


def volume_to_slices(volume: np.ndarray, patient: dict, output_dir: Path, patient_id: str) -> int:
    """Crop and resize a label volume like slice_patient() does for the ground truth; returns the slice count."""
    if list(volume.shape) != patient["original_shape"]:
        raise ValueError(f"{patient_id}: prediction shape {list(volume.shape)} "
                         f"!= original CT shape {patient['original_shape']}")
    x0, x1, y0, y1 = patient["crop_bounds"]
    cropped = volume[x0:x1, y0:y1]
    shape = tuple(patient["output_shape"][:2])

    for z_index in range(cropped.shape[2]):
        label_slice = _resize_label(cropped[:, :, z_index], shape)
        if not set(np.unique(label_slice).astype(int)).issubset(SEGTHOR_LABELS):
            raise ValueError(f"{patient_id}: unexpected labels in slice {z_index}")
        Image.fromarray((label_slice * LABEL_PNG_SCALE).astype(np.uint8)).save(
            output_dir / f"{patient_id}_{z_index:04d}.png"
        )
    return cropped.shape[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Folder with Patient_XX*.nii.gz predictions in the original CT space.")
    parser.add_argument("--preprocessing-json", type=Path, required=True,
                        help="preprocessing.json of the preprocessed dataset the ground truth comes from.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.preprocessing_json.read_text(encoding="utf-8"))
    patients = {patient["patient"]: patient for patient in metadata["patients"]}

    prediction_paths = sorted(args.pred_dir.glob("Patient_*.nii*"))
    if not prediction_paths:
        raise RuntimeError(f"No Patient_*.nii* predictions in {args.pred_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in prediction_paths:
        patient_id = re.match(r"Patient_\d+", path.name).group(0)
        image = nib.load(str(path))
        orientation = "".join(nib.aff2axcodes(image.affine))
        if orientation != patients[patient_id]["orientation"]:
            raise ValueError(f"{patient_id}: prediction orientation {orientation} "
                             f"!= CT orientation {patients[patient_id]['orientation']}")
        volume = np.rint(np.asanyarray(image.dataobj)).astype(np.uint8)
        slices = volume_to_slices(volume, patients[patient_id], args.output_dir, patient_id)
        print(f"{patient_id}: {slices} slices")


if __name__ == "__main__":
    main()
