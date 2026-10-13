from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize


def load_nifti_labels(
    path: Path,
    classes: int,
) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    """Load and validate a 3D integer label volume."""
    image = nib.load(str(path))
    labels = np.asanyarray(image.dataobj)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3D prediction at {path}, got {labels.shape}")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
        raise ValueError(f"Prediction labels must be finite integers in {path}")
    if labels.size and (labels.min() < 0 or labels.max() >= classes):
        raise ValueError(
            f"Prediction labels in {path} must be between 0 and {classes - 1}; "
            f"found {np.unique(labels)}"
        )
    return labels.astype(np.uint8), image


def slices_to_reference_volume(
    slices_zyx: np.ndarray,
    reference_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    """Resize decoded axial prediction slices and arrange them like a NIfTI XYZ volume."""
    if slices_zyx.ndim != 3:
        raise ValueError(f"Expected a 3D slice stack, got shape {slices_zyx.shape}")
    if slices_zyx.shape[0] != reference_shape_xyz[2]:
        raise ValueError(
            f"Prediction has {slices_zyx.shape[0]} slices but reference has "
            f"{reference_shape_xyz[2]}; cannot match slice indices safely"
        )
    if slices_zyx.shape[1:] == reference_shape_xyz[:2]:
        resized_zyx = slices_zyx
    else:
        resized_zyx = np.stack(
            [
                resize(
                    slice_labels,
                    reference_shape_xyz[:2],
                    order=0,
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)
                for slice_labels in slices_zyx
            ],
            axis=0,
        )
    return np.transpose(resized_zyx, (1, 2, 0))


def save_nifti_labels(
    labels_xyz: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = nib.Nifti1Image(
        labels_xyz.astype(np.uint8),
        reference.affine,
        reference.header.copy(),
    )
    output.set_data_dtype(np.uint8)
    nib.save(output, str(output_path))
