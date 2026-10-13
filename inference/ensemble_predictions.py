#!/usr/bin/env python3
"""Majority-vote ensemble of the cross-validation fold models' predictions.

Two kinds of input, one folder per fold, patients taken from the folders:
  --kind slices: 2D/2.5D PNG slice predictions (inference/predict_segthor_from_weights.py). The vote is taken on
                 the slices and saved as PNG slices on the same grid; with --preprocessing-json (from
                 inference/prepare_test_slices.py) the voted slices are instead mapped back to the original CT
                 (resized to the crop, nearest neighbour, placed at the crop) and saved as Patient_XX.nii.gz.
  --kind nifti:  3D predictions already in the original CT space (MONAI bundle inference.yaml, *_seg.nii.gz),
                 saved as Patient_XX.nii.gz with the same affine and header.

Each voxel gets the class most folds predict; a tie goes to the lower class index (background first).

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


def patients_in(fold_dir: Path, kind: str) -> dict[str, list[Path]]:
    """patient -> its prediction files in one fold folder (sorted slices, or the one volume)."""
    pattern = "Patient_*.png" if kind == "slices" else "Patient_*.nii*"
    found: dict[str, list[Path]] = {}
    for path in sorted(fold_dir.glob(pattern)):
        found.setdefault(re.match(r"Patient_\d+", path.name).group(0), []).append(path)
    if not found:
        raise RuntimeError(f"No {pattern} predictions in {fold_dir}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kind", choices=["slices", "nifti"], required=True)
    parser.add_argument("--fold-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--preprocessing-json", type=Path,
                        help="--kind slices: preprocessing.json of inference/prepare_test_slices.py (crops and CT paths) "
                             "to stitch the voted slices into NIfTI volumes in the CT space. Without it the voted "
                             "slices are saved as PNGs (labels x 63), on the same grid as the input slices.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    folds = [patients_in(d, args.kind) for d in args.fold_dirs]
    for d, fold in zip(args.fold_dirs[1:], folds[1:]):
        if {p: len(f) for p, f in fold.items()} != {p: len(f) for p, f in folds[0].items()}:
            raise RuntimeError(f"{d} does not have the same patients/slices as {args.fold_dirs[0]}")
    ct_info = ({p["patient"]: p for p in json.loads(args.preprocessing_json.read_text(encoding="utf-8"))["patients"]}
               if args.preprocessing_json else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pid, files in folds[0].items():
        if args.kind == "slices":
            voted = vote([load_slices(d, pid, len(files)) for d in args.fold_dirs])
            if ct_info is None:  # raw voted slices
                for z in range(voted.shape[2]):
                    output = args.output_dir / f"{pid}_{z:04d}.png"
                    if output.exists():
                        raise RuntimeError(f"Refusing to overwrite {output}")
                    Image.fromarray(voted[:, :, z] * LABEL_PNG_SCALE).save(output)
                labels = voted
            else:
                labels = slices_to_ct_space(voted, ct_info[pid])
                save_like_ct(labels, Path(ct_info[pid]["ct_path"]), args.output_dir / f"{pid}.nii.gz")
        else:
            if any(len(fold[pid]) != 1 for fold in folds):
                raise RuntimeError(f"Expected one prediction volume for {pid} in every fold folder")
            labels = vote([np.rint(np.asanyarray(nib.load(str(fold[pid][0])).dataobj)).astype(np.uint8)
                           for fold in folds])
            reference = Path(ct_info[pid]["ct_path"]) if ct_info else folds[0][pid][0]  # fold predictions share the CT's grid
            output = args.output_dir / f"{pid}.nii.gz"
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite {output}")
            save_like_ct(labels, reference, output)
        print(f"{pid}: " + ", ".join(f"{k}:{int((labels == k).sum())}" for k in range(1, CLASSES)) + " voxels")


if __name__ == "__main__":
    main()
