#!/usr/bin/env python3
"""Majority-vote ensemble of the cross-validation fold models' predictions, saved as NIfTI in the CT's space.

Two kinds of input, one folder per fold:
  --kind slices: 2D/2.5D PNG slice predictions (inference/predict_segthor_from_weights.py) on the grid of
                 inference/prepare_test_slices.py. The vote is taken on the slices, which are then mapped back to
                 the original CT: resized to the crop (nearest neighbour) and placed at the crop, background outside.
  --kind nifti:  3D predictions already in the original CT space (MONAI bundle inference.yaml, *_seg.nii.gz).

Each voxel gets the class most folds predict; a tie goes to the lower class index (background first).
Output: <output-dir>/Patient_XX.nii.gz with the CT's affine and header, uint8 labels 0-4.

Usage:
    python -m inference.ensemble_predictions --kind slices --fold-dirs run/fold_0 ... run/fold_4 \
        --preprocessing-json data/SEGTHOR_TEST_PREPROCESSED/preprocessing.json --output-dir run/ensemble
"""

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.transform import resize

CLASSES = 5
LABEL_PNG_SCALE = 63


def vote(predictions: list[np.ndarray]) -> np.ndarray:
    counts = np.stack([(np.stack(predictions) == k).sum(axis=0) for k in range(CLASSES)])
    return counts.argmax(axis=0).astype(np.uint8)  # argmax returns the first maximum: ties go to the lower class


def load_slices(fold_dir: Path, patient: str, slices: int) -> np.ndarray:
    volume = np.stack([np.asarray(Image.open(fold_dir / f"{patient}_{z:04d}.png")) for z in range(slices)], axis=-1)
    labels = np.rint(volume / LABEL_PNG_SCALE).astype(np.uint8)
    if labels.max() >= CLASSES:
        raise ValueError(f"Unexpected labels for {patient} in {fold_dir}")
    return labels


def slices_to_ct_space(labels: np.ndarray, patient: dict) -> np.ndarray:
    """Undo the resize and crop of inference/prepare_test_slices.py."""
    x0, x1, y0, y1 = patient["crop_bounds"]
    crop = resize(labels, (x1 - x0, y1 - y0, labels.shape[2]), order=0, mode="edge",
                  preserve_range=True, anti_aliasing=False).astype(np.uint8)
    volume = np.zeros(patient["original_shape"], dtype=np.uint8)
    volume[x0:x1, y0:y1] = crop
    return volume


def save_like_ct(labels: np.ndarray, ct_path: Path, output: Path) -> None:
    ct = nib.load(str(ct_path))
    if labels.shape != ct.shape:
        raise ValueError(f"Prediction shape {labels.shape} != CT shape {ct.shape} for {ct_path.name}")
    header = ct.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(labels, ct.affine, header), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kind", choices=["slices", "nifti"], required=True)
    parser.add_argument("--fold-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--preprocessing-json", type=Path, required=True,
                        help="preprocessing.json of inference/prepare_test_slices.py: crops and CT paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    patients = json.loads(args.preprocessing_json.read_text(encoding="utf-8"))["patients"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for patient in patients:
        pid = patient["patient"]
        output = args.output_dir / f"{pid}.nii.gz"
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite {output}")
        if args.kind == "slices":
            labels = slices_to_ct_space(vote([load_slices(d, pid, patient["output_shape"][2])
                                              for d in args.fold_dirs]), patient)
        else:
            folds = []
            for d in args.fold_dirs:
                matches = [p for p in d.glob(f"{pid}*.nii*") if re.match(rf"{pid}(\D|$)", p.name)]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one prediction for {pid} in {d}, found {matches}")
                folds.append(np.rint(np.asanyarray(nib.load(str(matches[0])).dataobj)).astype(np.uint8))
            labels = vote(folds)
        save_like_ct(labels, Path(patient["ct_path"]), output)
        print(f"{pid}: " + ", ".join(f"{k}:{int((labels == k).sum())}" for k in range(1, CLASSES)) + " voxels")


if __name__ == "__main__":
    main()
