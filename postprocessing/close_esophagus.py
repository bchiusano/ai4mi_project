import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing

from postprocessing.connected_compontents import load_volumes, save_volumes


def close_class_along_z(volume, class_index, z_len=5, xy_size=3):
    mask = volume == class_index
    padded = np.pad(mask, ((z_len, z_len), (0, 0), (0, 0)), mode="edge")
    structure = np.ones((z_len, xy_size, xy_size), dtype=bool)
    closed = binary_closing(padded, structure=structure)[z_len:-z_len]
    result = volume.copy()
    result[closed & (volume == 0)] = class_index
    return result


classes, label_step = 5, 63.0
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
z_len, xy_size = int(sys.argv[3]), int(sys.argv[4])

volumes, paths = load_volumes(src, classes, label_step)
processed = {p: close_class_along_z(v, 1, z_len, xy_size) for p, v in volumes.items()}
save_volumes(processed, paths, dst, label_step)
print(f"Saved {len(processed)} patients to {dst} (z_len={z_len}, xy_size={xy_size})")