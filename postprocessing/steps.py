"""Postprocessing steps for label volumes ordered (z, a, b), axis 0 = slice index.

Only NumPy and SciPy are needed, so these functions can be tested without the project.
"""

import numpy as np
from scipy.ndimage import (
    binary_closing,
    center_of_mass,
    distance_transform_edt,
    generate_binary_structure,
    label,
    shift,
)

CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


def keep_largest_component(volume: np.ndarray, class_index: int, connectivity: int = 6) -> np.ndarray:
    """Keep the largest 3D component of one class; the other components become background."""
    structure = generate_binary_structure(3, CONNECTIVITY_RANK[connectivity])
    components, count = label(volume == class_index, structure=structure)
    if count <= 1:
        return volume.copy()
    sizes = np.bincount(components.ravel())[1:]
    keep = int(np.argmax(sizes)) + 1
    result = volume.copy()
    result[(volume == class_index) & (components != keep)] = 0
    return result


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(mask) - distance_transform_edt(~mask)


def bridge_gaps(volume: np.ndarray, class_index: int, min_size: int, max_gap: int) -> np.ndarray:
    """Fill the empty slices between consecutive pieces of one class.

    Pieces smaller than ``min_size`` voxels are ignored. For each gap of 1..max_gap slices,
    the cross-sections at the two ends are blended (as signed distance maps) while their
    centres move from one end to the other, so sideways offsets are handled. Only
    background voxels are filled.
    """
    components, count = label(volume == class_index)
    pieces = []
    for index in range(1, count + 1):
        mask = components == index
        if mask.sum() >= min_size:
            z = np.flatnonzero(mask.any(axis=(1, 2)))
            pieces.append((int(z[0]), int(z[-1]), mask))
    pieces.sort(key=lambda piece: piece[0])

    result = volume.copy()
    if len(pieces) < 2:
        return result

    previous = pieces[0]
    for piece in pieces[1:]:
        z_a, z_b = previous[1], piece[0]  # last slice above the gap, first slice below it
        if 1 <= z_b - z_a - 1 <= max_gap:
            mask_a, mask_b = previous[2][z_a], piece[2][z_b]
            centre_a = np.array(center_of_mass(mask_a))
            centre_b = np.array(center_of_mass(mask_b))
            distance_a, distance_b = _signed_distance(mask_a), _signed_distance(mask_b)
            for z in range(z_a + 1, z_b):
                t = (z - z_a) / (z_b - z_a)
                centre = (1 - t) * centre_a + t * centre_b
                moved_a = shift(distance_a, centre - centre_a, order=1, mode="nearest")
                moved_b = shift(distance_b, centre - centre_b, order=1, mode="nearest")
                filled = ((1 - t) * moved_a + t * moved_b) > 0
                result[z][filled & (volume[z] == 0)] = class_index
        if piece[1] > previous[1]:  # pieces that overlap in z: keep the one reaching further
            previous = piece
    return result


def close_along_z(volume: np.ndarray, class_index: int, z_len: int, xy_size: int) -> np.ndarray:
    """Morphological closing with a (z_len, xy_size, xy_size) box; only fills background voxels."""
    mask = volume == class_index
    if not mask.any():
        return volume.copy()
    padded = np.pad(mask, ((z_len, z_len), (0, 0), (0, 0)), mode="edge")
    structure = np.ones((z_len, xy_size, xy_size), dtype=bool)
    closed = binary_closing(padded, structure=structure)[z_len:-z_len]
    result = volume.copy()
    result[closed & (volume == 0)] = class_index
    return result


def process_volume(
    volume: np.ndarray,
    lcc_classes=(),
    bridge_classes=(),
    bridge_min_size: int = 200,
    bridge_max_gap: int = 30,
    close_classes=(),
    close_z: int = 5,
    close_xy: int = 3,
):
    """Run largest-component, then gap bridging, then closing. Returns (volume, changed-voxel log)."""
    log: dict[str, int] = {}
    current = volume

    def apply(name, function, *arguments):
        nonlocal current
        updated = function(current, *arguments)
        log[name] = int((updated != current).sum())
        current = updated

    for class_index in lcc_classes:
        apply(f"largest_component_{class_index}", keep_largest_component, class_index)
    for class_index in bridge_classes:
        apply(f"bridge_{class_index}", bridge_gaps, class_index, bridge_min_size, bridge_max_gap)
    for class_index in close_classes:
        apply(f"close_{class_index}", close_along_z, class_index, close_z, close_xy)
    return current, log