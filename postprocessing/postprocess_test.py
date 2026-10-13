#!/usr/bin/env python3
"""Postprocess held-out PNG slices or NIfTI predictions and save one volume per scan."""

import argparse
import sys
from pathlib import Path

import nibabel as nib

if __package__:
    from .connected_compontents import keep_largest_components, load_volumes
    from .nifti_utils import (
        load_nifti_labels,
        save_nifti_labels,
        slices_to_reference_volume,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from postprocessing.connected_compontents import keep_largest_components, load_volumes
    from postprocessing.nifti_utils import (
        load_nifti_labels,
        save_nifti_labels,
        slices_to_reference_volume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/heldout_predictions"),
        help="Root containing one directory of PNG slices or NIfTI predictions per model.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/segthor_train_full/train"),
        help="Root containing <Patient_**>/<Patient_**>.nii.gz source scans for geometry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/heldout_predictions_postprocessed"),
        help="Destination for postprocessed .nii.gz volumes, grouped by model.",
    )
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--label-step", type=float, default=63.0)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    args = parser.parse_args()

    if args.classes < 2 or args.classes > 256:
        parser.error("--classes must be between 2 and 256")
    if args.label_step <= 0:
        parser.error("--label-step must be positive")
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist or is not a directory: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from --input-dir")

    model_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    models_processed = 0
    for model_dir in model_dirs:
        png_paths = sorted(model_dir.glob("*.png"))
        nifti_paths = sorted(
            [*model_dir.glob("*.nii"), *model_dir.glob("*.nii.gz")]
        )
        if png_paths:
            if not args.reference_dir.is_dir():
                parser.error(
                    f"--reference-dir does not exist or is not a directory: "
                    f"{args.reference_dir}"
                )
            predictions, _ = load_volumes(model_dir, args.classes, args.label_step)
            prediction_images = {}
            nifti_input = False
        elif nifti_paths:
            predictions = {}
            prediction_images = {}
            for prediction_path in nifti_paths:
                stem = (
                    prediction_path.name[:-7]
                    if prediction_path.name.endswith(".nii.gz")
                    else prediction_path.stem
                )
                patient_id = stem
                if patient_id in predictions:
                    raise ValueError(
                        f"Multiple NIfTI predictions found for "
                        f"{model_dir.name}/{patient_id}"
                    )
                prediction, prediction_image = load_nifti_labels(
                    prediction_path, args.classes
                )
                predictions[patient_id] = prediction
                prediction_images[patient_id] = prediction_image
            nifti_input = True
        else:
            print(f"Skipping {model_dir.name}: no PNG slices or NIfTI predictions")
            continue

        for patient_id, prediction_zyx in sorted(predictions.items()):
            if nifti_input:
                reference = prediction_images[patient_id]
            else:
                reference_path = args.reference_dir / patient_id / f"{patient_id}.nii.gz"
                if not reference_path.is_file():
                    raise FileNotFoundError(
                        f"No source scan found for {model_dir.name}/{patient_id}: "
                        f"expected {reference_path}"
                    )
                reference = nib.load(str(reference_path))
            if len(reference.shape) != 3:
                raise ValueError(
                    f"Expected 3D reference for {model_dir.name}/{patient_id}, "
                    f"got {reference.shape}"
                )
            filtered_zyx = keep_largest_components(
                prediction_zyx,
                args.classes,
                args.connectivity,
            )
            labels_xyz = (
                filtered_zyx
                if nifti_input
                else slices_to_reference_volume(
                    filtered_zyx,
                    tuple(int(size) for size in reference.shape),
                )
            )
            output_path = args.output_dir / model_dir.name / f"{patient_id}.nii.gz"
            save_nifti_labels(labels_xyz, reference, output_path)
            print(f"Saved {model_dir.name}/{patient_id}: {output_path}")
        models_processed += 1

    if models_processed == 0:
        raise RuntimeError(f"No model prediction PNGs or NIfTIs found under {args.input_dir}")


if __name__ == "__main__":
    main()
