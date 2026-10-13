import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, label, shift

from postprocessing.connected_compontents import load_volumes, save_volumes


def signed_distance(mask):
    return distance_transform_edt(mask) - distance_transform_edt(~mask)


def bridge_class(volume, class_index, min_size, max_gap):
    comp, n = label(volume == class_index)
    pieces = []
    for i in range(1, n + 1):
        c = comp == i
        if c.sum() >= min_size:
            z = np.where(c.any(axis=(1, 2)))[0]
            pieces.append((z.min(), z.max(), c))
    pieces.sort(key=lambda p: p[0])

    result = volume.copy()
    for (_, za, a), (zb, _, b) in zip(pieces, pieces[1:]):
        gap = zb - za - 1
        if gap < 1 or gap > max_gap:
            continue
        ma, mb = a[za], b[zb]
        ca, cb = np.array(center_of_mass(ma)), np.array(center_of_mass(mb))
        sda, sdb = signed_distance(ma), signed_distance(mb)
        for z in range(za + 1, zb):
            t = (z - za) / (zb - za)
            ct = (1 - t) * ca + t * cb
            sa = shift(sda, ct - ca, order=1, mode="nearest")
            sb = shift(sdb, ct - cb, order=1, mode="nearest")
            filled = ((1 - t) * sa + t * sb) > 0
            result[z][filled & (volume[z] == 0)] = class_index
    return result


classes, label_step = 5, 63.0
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
min_size, max_gap = int(sys.argv[3]), int(sys.argv[4])

volumes, paths = load_volumes(src, classes, label_step)
processed = {}
for patient, volume in volumes.items():
    processed[patient] = bridge_class(volume, 1, min_size, max_gap)
    print(f"{patient}: added {(processed[patient] != volume).sum()} voxels")
save_volumes(processed, paths, dst, label_step)