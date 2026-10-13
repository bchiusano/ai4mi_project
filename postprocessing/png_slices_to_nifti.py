import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize

from postprocessing.connected_compontents import load_volumes

classes, label_step = 5, 63.0
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
prep, ref_root = Path(sys.argv[3]), Path(sys.argv[4])

meta = {p["patient"]: p for p in json.loads(prep.read_text())["patients"]}
volumes, _ = load_volumes(src, classes, label_step)  # (z, rows, cols)
dst.mkdir(parents=True, exist_ok=True)

for patient, vol in volumes.items():
    info = meta[patient]
    X, Y, Z = info["original_shape"]
    x0, x1, y0, y1 = info["crop_bounds"]
    if vol.shape[0] != Z:
        raise ValueError(f"{patient}: {vol.shape[0]} slices but CT has {Z}")
    out = np.zeros((X, Y, Z), dtype=np.uint8)
    for z in range(Z):
        out[x0:x1, y0:y1, z] = resize(
            vol[z], (x1 - x0, y1 - y0), order=0, mode="constant",
            preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)
    ref = nib.load(str(ref_root / patient / "GT.nii.gz"))
    img = nib.Nifti1Image(out, ref.affine, ref.header.copy())
    img.set_data_dtype(np.uint8)
    nib.save(img, str(dst / f"{patient}.nii.gz"))
    print(f"Saved {patient}")