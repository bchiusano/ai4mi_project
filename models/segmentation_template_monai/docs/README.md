
# SegTHOR SwinUNETR Bundle

3D segmentation of the SegTHOR organs at risk (esophagus, heart, trachea, aorta) with a SwinUNETR fine-tuned from the
[MONAI BTCV bundle](https://huggingface.co/MONAI/swin_unetr_btcv_segmentation). Adapted from the MONAI model zoo
segmentation template.

## Setup

The pretrained BTCV weights (`models/btcv_swin_unetr.pt`, not in git) are listed in `large_files.yml`:

```sh
python -m monai.bundle download_large_files --bundle_path models/segmentation_template_monai
```

The 3D volumes come from `data/segthor_train_full.zip`, unzipped so that `data_root` holds one `Patient_XX/` folder
per patient with `Patient_XX.nii.gz` and `GT.nii.gz` (`data/segthor_train_full/train` by default).

## Configs

All configs are used on top of `configs/common.yaml`, which defines the network, classes, spacing, intensity window
and sliding window inferer:

- `train.yaml`: trains on the train patients of one fold of `data/SEGTHOR_FULL_PREPROCESSED_split_cv/folds.json`
  (the same split as the 2D runs) and validates on its val patients. Saves the best model by validation Dice as
  `model.pt` in `output_dir`. The pretrained weights are loaded by `scripts/pretrained.py`, which also initialises
  the esophagus and aorta outputs from the matching BTCV classes; `--pretrained_path None` trains from scratch.
- `test.yaml`: evaluates a checkpoint on the val patients of a fold. Predictions are mapped back to the original CT
  space before computing Dice against the unmodified GT; per-patient, per-organ Dice goes to
  `val_mean_dice_raw.csv`.
- `inference.yaml`: segments every `Patient_*.nii*` under `dataset_dir` and saves `Patient_XX_seg.nii.gz` files in
  the original CT space.
- `multi_gpu_train.yaml`: mixin for DDP training with `torchrun`, see `train_multigpu.sh`.

Any config value can be overridden on the command line, e.g. `--fold 2` or `--hu_window "[-1000, 1000]"`.

## Running

On Snellius, use the job files: `snellius/train_swin_unetr.sbatch` trains and evaluates one fold,
`snellius/submit_swin_unetr_cv.sh` submits all five. Locally, the scripts in `docs/` run each config with the project
venv:

```sh
models/segmentation_template_monai/docs/train.sh --fold 0
models/segmentation_template_monai/docs/test.sh --fold 0 --ckpt_path <run dir>/model.pt
models/segmentation_template_monai/docs/inference.sh --ckpt_path <run dir>/model.pt --dataset_dir <dir>
```
