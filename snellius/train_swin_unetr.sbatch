#!/usr/bin/env bash
#SBATCH --job-name=segthor_swin_unetr
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fine-tunes the pretrained MONAI BTCV SwinUNETR (bundle in models/segmentation_template_monai) on one
# cross-validation fold of data/SEGTHOR_FULL_PREPROCESSED_split_cv/folds.json, the same patient split as the 2D runs,
# using the 3D volumes from data/segthor_train_full.zip. The best checkpoint (by validation Dice) is then evaluated on the fold's val
# patients in the original CT space.
# Submit all folds with snellius/submit_swin_unetr_cv.sh. Quick end-to-end check of the pipeline:
#   sbatch --time=00:30:00 --export=ALL,FOLD=0,RUN_TAG=smoke,EPOCHS=2,VAL_INTERVAL=1 snellius/train_swin_unetr.sbatch

set -euo pipefail

: "${FOLD:?Submit with FOLD=<fold index 0-4>}"
RUN_TAG="${RUN_TAG:-cv}"
EPOCHS="${EPOCHS:-500}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"  # Epochs between validations
PATIENCE="${PATIENCE:-10}"  # Validations without validation Dice improvement before stopping

PROJECT="$HOME/ai4mi_project"
BUNDLE="$PROJECT/models/segmentation_template_monai"
DATA_ZIP="$PROJECT/data/segthor_train_full.zip"
PRETRAINED="$BUNDLE/models/btcv_swin_unetr.pt"
# The volumes are unzipped to the node-local disk: fast to read and no clashes between the fold jobs
DATA_DIR="${TMPDIR:?TMPDIR is not set}/segthor_train_full"
RESULTS_DIR="$PROJECT/results/segthor_swin_unetr/$RUN_TAG/fold_$FOLD"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1  # the data loader workers already use all CPUs
export PYTHONPATH="$BUNDLE"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Fold: $FOLD, epochs: $EPOCHS, validation every $VAL_INTERVAL epochs, early stopping patience: $PATIENCE"
echo "Results directory: $RESULTS_DIR"
echo "Git commit: $(git rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch, monai, ignite, einops, fire; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch CUDA:", torch.cuda.get_device_name(0), "| MONAI", monai.__version__)'

if [[ ! -f "$PRETRAINED" ]]; then
    echo "Pretrained weights missing: $PRETRAINED" \
         "(fetch with: python -m monai.bundle download_large_files --bundle_path $BUNDLE)" >&2
    exit 1
fi
if [[ -e "$RESULTS_DIR" ]]; then
    echo "Directory already exists; refusing to overwrite: $RESULTS_DIR" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR" "$DATA_DIR"

unzip -q "$DATA_ZIP" -d "$DATA_DIR"
volume_count=$(find "$DATA_DIR/train" -name 'GT.nii.gz' | wc -l)
echo "Volumes: $volume_count in $DATA_DIR/train"
if [[ "$volume_count" -ne 40 ]]; then
    echo "Expected 40 labelled volumes in $DATA_ZIP" >&2
    exit 1
fi

bundle_args=(
    --meta_file "$BUNDLE/configs/metadata.json"
    --logging_file "$BUNDLE/configs/logging.conf"
    --bundle_root "$BUNDLE"
    --data_root "$DATA_DIR/train"
    --fold "$FOLD"
)

python -u -m monai.bundle run \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/train.yaml']" \
    "${bundle_args[@]}" \
    --output_dir "$RESULTS_DIR" \
    --pretrained_path "$PRETRAINED" \
    --num_epochs "$EPOCHS" \
    --val_interval "$VAL_INTERVAL" \
    --early_stop_patience "$PATIENCE" \
    --num_workers $((SLURM_CPUS_PER_TASK - 2))

# Per-patient, per-organ Dice of the best model on the fold's val patients, plus the predicted segmentations
python -u -m monai.bundle run \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/test.yaml']" \
    "${bundle_args[@]}" \
    --ckpt_path "$RESULTS_DIR/model.pt" \
    --output_dir "$RESULTS_DIR/eval" \
    --save_pred True \
    | tee "$RESULTS_DIR/eval_summary.txt"

echo "Finished: $(date)"
