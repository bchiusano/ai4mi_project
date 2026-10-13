import sys, glob
import numpy as np
from PIL import Image
from scipy.ndimage import label

def vol(folder, patient):
    files = sorted(glob.glob(f"{folder}/{patient}_*.png"))
    return np.stack([np.rint(np.asarray(Image.open(f)) / 63).astype(np.uint8) for f in files])

patient = sys.argv[1]
pred = vol("results/heldout_predictions/ce_dice_25d", patient) == 1
gt = vol("data/SEGTHOR_FULL/gt", patient) == 1

zg = np.where(gt.any(axis=(1, 2)))[0]
print(f"{patient} GT: {gt.sum()} voxels, z {zg.min()}-{zg.max()}")
comp, n = label(pred)
for i in range(1, n + 1):
    c = comp == i
    z = np.where(c.any(axis=(1, 2)))[0]
    print(f"  pred component {i}: {c.sum()} voxels, z {z.min()}-{z.max()}, overlap with GT {(c & gt).sum()}")