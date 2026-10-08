# Final SegTHOR preprocessing pipeline

The official clean 40-patient SegTHOR archive is converted into one reusable,
architecture-independent 2-D dataset.

## Selected processing

For every patient, the pipeline:

1. verifies that CT and ground truth have identical shapes and affine matrices,
   LPS orientation, finite CT values, and labels limited to `0..4`;
2. clips CT intensities to `[-1000, 1000]` HU and maps them to uint8 `[0,255]`;
3. takes a centred `384 x 384 mm` in-plane field of view and stops if this would
   remove any labelled foreground voxel;
4. resamples each axial CT slice to `384 x 384` with bilinear interpolation and
   anti-aliasing;
5. resamples masks with nearest-neighbour interpolation, preserving categorical
   labels;
6. preserves the number of slices and original z-spacing; and
7. saves the effective 3-D spacing and all transformation metadata.

CLAHE and online augmentation are intentionally excluded from the final
pipeline because the validation ablations did not show a reliable improvement.

## Fixed patient split

With split seed `0`, the output contains 28 training, 6 validation, and 6
labelled test patients. Patients 1–20 had already been exposed during preliminary
work, so the test set is sampled only from Patients 21–40. The test set must
remain unused until final model evaluation.

All slices are stored once in `img/` and `gt/`; `split.json` assigns patients to
subsets. `data_loading.dataset.SliceDataset` reads this manifest automatically.

## Generate locally

From the repository root:

```bash
python -m preprocessing.slice_segthor \
  --source_dir data/segthor_train_full \
  --dest_dir data/SEGTHOR_FULL \
  --shape 384 384 \
  --target-spacing 1.0 1.0 \
  --window -1000 1000 \
  --validation-count 6 \
  --test-count 6 \
  --test-pool-start 21 \
  --split-seed 0 \
  --process 1

python -m preprocessing.validate_processed_segthor data/SEGTHOR_FULL
```

Increase `--process` to a safe number of CPU workers for faster conversion, or
use `-1` for every available CPU. The destination must not already exist; this
prevents accidental partial overwrites.

## Use the shared preprocessed archive

Place `SEGTHOR_FULL_preprocessed.zip` in `data/` and verify its SHA-256 against
`data/SEGTHOR_FULL_preprocessed.sha256`. Then extract it from the repository
root:

```bash
unzip data/SEGTHOR_FULL_preprocessed.zip -d data
python -m preprocessing.validate_processed_segthor data/SEGTHOR_FULL
```

PowerShell users can extract it with:

```powershell
Expand-Archive -LiteralPath data\SEGTHOR_FULL_preprocessed.zip -DestinationPath data
python -m preprocessing.validate_processed_segthor data\SEGTHOR_FULL
```

## Output

```text
data/SEGTHOR_FULL/
├── img/                 # uint8 CT slices
├── gt/                  # encoded masks: 0, 63, 126, 189, 252
├── split.json           # fixed train/validation/test patients
├── spacing.pkl          # effective x/y spacing and original z-spacing
└── preprocessing.json   # complete per-patient transformation metadata
```

Mask values correspond to background, esophagus, heart, trachea, and aorta.

## Use in training

The existing loader selects patients from `split.json`:

```python
from data_loading.dataset import SliceDataset

train_set = SliceDataset("train", "data/SEGTHOR_FULL", img_transform, gt_transform)
val_set = SliceDataset("val", "data/SEGTHOR_FULL", img_transform, gt_transform)
test_set = SliceDataset("test", "data/SEGTHOR_FULL", img_transform, gt_transform)
```

The reference training command is:

```bash
python main.py --dataset SEGTHOR_FULL --mode full --loss ce_dice \
  --seed 0 --epochs 25 --dest results/segthor_full/seed0 --gpu
```
