#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

if __package__:
    from .connected_compontents import keep_largest_component
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from postprocessing.connected_compontents import keep_largest_component


def postprocess_prediction(
    input_path: Path,
    output_path: Path,
    classes: int,
    connectivity: int,
) -> None:
    image = nib.load(str(input_path))
    labels = np.asanyarray(image.dataobj)
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D prediction in {input_path}, got shape {labels.shape}")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
        raise ValueError(f"Prediction contains non-finite or non-integer labels: {input_path}")
    if labels.size and (labels.min() < 0 or labels.max() >= classes):
        raise ValueError(
            f"Labels in {input_path} must be between 0 and {classes - 1}; "
            f"found {np.unique(labels)}"
        )

    processed = labels.astype(np.uint8)
    for class_index in range(1, classes):
        processed = keep_largest_component(processed, class_index, connectivity)

    output_image = nib.Nifti1Image(
        processed,
        image.affine,
        header=image.header.copy(),
    )
    output_image.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, str(output_path))


def nifti_files(model_dir: Path) -> list[Path]:
    return sorted(
        path for path in model_dir.iterdir()
        if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply 3D largest-connected-component filtering to test predictions "
            "without calculating evaluation metrics."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/test_predictions_ensamble"),
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test_predictions_ensamble_postprocessed"),
        help="Destination for postprocessed NIfTI volumes, grouped by model.",
    )
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    args = parser.parse_args()

    if args.classes < 2 or args.classes > 256:
        parser.error("--classes must be between 2 and 256")
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist or is not a directory: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from --input-dir")

    model_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not model_dirs:
        raise RuntimeError(f"No model directories found in {args.input_dir}")

    for model_dir in model_dirs:
        predictions = nifti_files(model_dir)
        if not predictions:
            raise RuntimeError(f"No NIfTI predictions found in {model_dir}")

        output_names = [
            path.name[:-7] + ".nii.gz" if path.name.endswith(".nii.gz")
            else path.stem + ".nii.gz"
            for path in predictions
        ]
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Multiple input NIfTI files map to the same output in {model_dir}")

        for input_path, output_name in zip(predictions, output_names):
            output_path = args.output_dir / model_dir.name / output_name
            postprocess_prediction(
                input_path,
                output_path,
                args.classes,
                args.connectivity,
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
