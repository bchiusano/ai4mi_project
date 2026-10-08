#!/usr/bin/env bash
#SBATCH --job-name=segthor_cv
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Trains one setting on one dataset folder with train/ and val/ (and optionally test/).
#  - Default: one cross-validation fold of the full 40-patient SegTHOR training set
#    (prepared by snellius/slice_split_segthor_full.sbatch), selected with FOLD.
#    Submit all settings and folds with snellius/submit_cv.sh.
#  - Any other folder: set DATA_DIR, RESULTS_NAME and RUN_SUBDIR, e.g. the fixed split
#    of SEGTHOR_FULL_PREPROCESSED (snellius/submit_preprocessed.sh).
# If DATA_DIR has a test/ folder, the best model is also evaluated on it.

set -euo pipefail

: "${SETTING:?Submit with SETTING=ce_2d, ce_25d, ce_dice_2d, ce_dice_25d or weighted_ce_25d}"
RUN_TAG="${RUN_TAG:-cv}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"  # Epochs without validation 3D dice improvement before stopping
SEED="${SEED:-0}"  # Training seed (weight init, shuffling, dropout)
LR_SCHEDULER="${LR_SCHEDULER:-plateau}"  # none or plateau (see main.py --lr-scheduler)
LR_PATIENCE="${LR_PATIENCE:-4}"  # Plateau scheduler only; keep below PATIENCE

case "$SETTING" in
    ce_2d)           LOSS=ce;      CONTEXT=0; CLASS_WEIGHTS=none ;;
    ce_25d)          LOSS=ce;      CONTEXT=1; CLASS_WEIGHTS=none ;;
    ce_dice_2d)      LOSS=ce_dice; CONTEXT=0; CLASS_WEIGHTS=none ;;
    ce_dice_25d)     LOSS=ce_dice; CONTEXT=1; CLASS_WEIGHTS=none ;;
    weighted_ce_25d) LOSS=ce;      CONTEXT=1; CLASS_WEIGHTS="${CLASS_WEIGHTS:-sqrt_inv}" ;;
    *)
        echo "Unsupported SETTING=$SETTING" >&2
        exit 2
        ;;
esac

PROJECT="$HOME/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
if [[ -z "${DATA_DIR:-}" ]]; then
    : "${FOLD:?Submit with FOLD=<fold index>, or with DATA_DIR=<dataset folder>}"
    DATA_DIR="$SCRATCH_PROJECT/data/SEGTHOR_FULL_CV/fold_$FOLD"
    RESULTS_NAME=segthor_full_cv
    RUN_SUBDIR="fold_$FOLD"
else
    : "${RESULTS_NAME:?Submit DATA_DIR together with RESULTS_NAME=<results folder name>}"
    RUN_SUBDIR="${RUN_SUBDIR:-split}"
fi
# Training writes to scratch: the per-epoch prediction folders (iterXXX, ~75k files per
# run) would exceed the home file-count quota. At the end, everything except those
# folders is copied to RESULTS_DIR in the project, which is what to keep.
RUN_DIR="$SCRATCH_PROJECT/results/$RESULTS_NAME/$RUN_TAG/$SETTING/$RUN_SUBDIR"
RESULTS_DIR="$PROJECT/results/$RESULTS_NAME/$RUN_TAG/$SETTING/$RUN_SUBDIR"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Setting: $SETTING (loss=$LOSS, class weights=$CLASS_WEIGHTS, context=$CONTEXT)"
echo "Data: $DATA_DIR"
echo "Seed: $SEED"
echo "Epochs: $EPOCHS, early stopping patience: $PATIENCE, lr scheduler: $LR_SCHEDULER (lr patience $LR_PATIENCE)"
echo "Run directory (scratch): $RUN_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Git commit: $(git rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch CUDA:", torch.cuda.get_device_name(0))'
python -m py_compile main.py utils.py losses.py \
    data_loading/dataset.py models/ENet.py models/ENet_25D.py models/ShallowNet.py \
    plotting/plot.py evaluation/summarize_segthor_run.py \
    evaluation/evaluate_saved_predictions.py inference/predict_segthor_from_weights.py
python -m unittest discover -s tests -v

if [[ ! -d "$DATA_DIR/train/img" || ! -d "$DATA_DIR/val/img" ]]; then
    echo "Dataset folder is missing train/ or val/: $DATA_DIR" \
         "(for the CV folds, run snellius/slice_split_segthor_full.sbatch first)" >&2
    exit 1
fi
# The dataset folders contain symlinks, so count them with -L
train_count=$(find -L "$DATA_DIR/train/img" -maxdepth 1 -type f -name '*.png' | wc -l)
val_count=$(find -L "$DATA_DIR/val/img" -maxdepth 1 -type f -name '*.png' | wc -l)
echo "Slices: train=$train_count val=$val_count"
if [[ "$train_count" -eq 0 || "$val_count" -eq 0 ]]; then
    echo "Empty train or val split: $DATA_DIR" >&2
    exit 1
fi

for d in "$RUN_DIR" "$RESULTS_DIR"; do
    if [[ -e "$d" ]]; then
        echo "Directory already exists; refusing to overwrite: $d" >&2
        exit 1
    fi
done
mkdir -p "$RUN_DIR"

# Copy the results to the project when the job ends, also if a later step fails,
# so a finished training is never left only on scratch
copy_results() {
    mkdir -p "$RESULTS_DIR"
    rsync -a --exclude 'iter[0-9][0-9][0-9]/' "$RUN_DIR/" "$RESULTS_DIR/"
    echo "Copied results to $RESULTS_DIR (per-epoch predictions stay in $RUN_DIR)"
}
trap copy_results EXIT

python -u main.py \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --seed "$SEED" \
    --lr-scheduler "$LR_SCHEDULER" \
    --lr-patience "$LR_PATIENCE" \
    --dataset SEGTHOR_FULL \
    --data-dir "$DATA_DIR" \
    --mode full \
    --loss "$LOSS" \
    --class-weights "$CLASS_WEIGHTS" \
    --context "$CONTEXT" \
    --dest "$RUN_DIR" \
    --gpu

python -m plotting.plot --metric_file "$RUN_DIR/loss_tra.npy" \
    --dest "$RUN_DIR/loss_train.png" --headless
python -m plotting.plot --metric_file "$RUN_DIR/loss_val.npy" \
    --dest "$RUN_DIR/loss_val.png" --headless
python -m plotting.plot --metric_file "$RUN_DIR/dice3d_val.npy" \
    --dest "$RUN_DIR/dice3d_val.png" --headless

python -m evaluation.summarize_segthor_run --run-dir "$RUN_DIR" | tee "$RUN_DIR/summary.txt"
python -m evaluation.evaluate_saved_predictions \
    --pred-dir "$RUN_DIR/best_epoch/val" \
    --gt-dir "$DATA_DIR/val/gt" \
    --csv "$RUN_DIR/best_epoch_3d_dice.csv" \
    | tee "$RUN_DIR/best_epoch_3d_dice.txt"

# Held-out test set: predict once with the best (validation-selected) weights
if [[ -d "$DATA_DIR/test/img" ]]; then
    python -m inference.predict_segthor_from_weights \
        --weights "$RUN_DIR/bestweights.pt" \
        --image-dir "$DATA_DIR/test/img" \
        --output-dir "$RUN_DIR/best_epoch/test" \
        --context "$CONTEXT" \
        --gpu
    python -m evaluation.evaluate_saved_predictions \
        --pred-dir "$RUN_DIR/best_epoch/test" \
        --gt-dir "$DATA_DIR/test/gt" \
        --csv "$RUN_DIR/test_3d_dice.csv" \
        | tee "$RUN_DIR/test_3d_dice.txt"
fi

echo "Finished: $(date)"
