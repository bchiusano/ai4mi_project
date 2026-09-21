#!/usr/bin/env bash
#SBATCH --job-name=segthor_ext
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${LOSS:?Submit with LOSS=dice or LOSS=ce_dice}"
RUN_TAG="${RUN_TAG:-preliminary_3d_dice}"

case "$LOSS" in
    dice|ce_dice) ;;
    *)
        echo "Unsupported LOSS=$LOSS; expected dice or ce_dice" >&2
        exit 2
        ;;
esac

PROJECT="$HOME/projects/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
EXPECTED_DATA="$SCRATCH_PROJECT/data/SEGTHOR_CLEAN_v2"
DATA_LINK="$PROJECT/data/SEGTHOR_CLEAN"
RUN_DIR="$SCRATCH_PROJECT/results/segthor_clean/$RUN_TAG/$LOSS"
RUN_PARENT="$SCRATCH_PROJECT/results/segthor_clean/$RUN_TAG"
GT_DIR="$DATA_LINK/val/gt"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Loss: $LOSS"
echo "Run directory: $RUN_DIR"
echo "Git commit: $(git rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch CUDA:", torch.cuda.get_device_name(0))'
python -m py_compile main.py utils.py losses.py plot.py summarize_segthor_run.py evaluate_saved_predictions.py
python -m unittest discover -s tests -v

if [[ ! -d "$EXPECTED_DATA/train/img" || ! -d "$EXPECTED_DATA/val/img" ]]; then
    echo "Verified corrected dataset is missing: $EXPECTED_DATA" >&2
    exit 1
fi

if [[ ! -e "$DATA_LINK" ]]; then
    ln -s "$EXPECTED_DATA" "$DATA_LINK"
fi

if [[ "$(readlink -f "$DATA_LINK")" != "$(readlink -f "$EXPECTED_DATA")" ]]; then
    echo "$DATA_LINK does not point to the verified final corrected dataset" >&2
    echo "Actual: $(readlink -f "$DATA_LINK")" >&2
    echo "Expected: $(readlink -f "$EXPECTED_DATA")" >&2
    exit 1
fi

train_count=$(find "$DATA_LINK/train/img" -maxdepth 1 -type f -name '*.png' | wc -l)
val_count=$(find "$DATA_LINK/val/img" -maxdepth 1 -type f -name '*.png' | wc -l)
if [[ "$train_count" -ne 2817 || "$val_count" -ne 915 ]]; then
    echo "Unexpected slice counts: train=$train_count val=$val_count" >&2
    exit 1
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

python -u main.py \
    --epochs 25 \
    --dataset SEGTHOR_CLEAN \
    --mode full \
    --loss "$LOSS" \
    --dest "$RUN_DIR" \
    --gpu

python plot.py --metric_file "$RUN_DIR/loss_tra.npy" \
    --dest "$RUN_DIR/loss_train.png" --headless
python plot.py --metric_file "$RUN_DIR/loss_val.npy" \
    --dest "$RUN_DIR/loss_val.png" --headless
python plot.py --metric_file "$RUN_DIR/dice3d_val.npy" \
    --dest "$RUN_DIR/dice3d_val.png" --headless

python summarize_segthor_run.py --run-dir "$RUN_DIR" | tee "$RUN_DIR/summary.txt"
python evaluate_saved_predictions.py \
    --pred-dir "$RUN_DIR/best_epoch/val" \
    --gt-dir "$GT_DIR" \
    --csv "$RUN_DIR/best_epoch_3d_dice.csv" \
    | tee "$RUN_DIR/best_epoch_3d_dice.txt"

if [[ "$LOSS" == "dice" ]]; then
    BASELINE_PRED="$SCRATCH_PROJECT/results/segthor_clean/baseline_patient19_fixed/best_epoch/val"
    if [[ -d "$BASELINE_PRED" ]]; then
        python evaluate_saved_predictions.py \
            --pred-dir "$BASELINE_PRED" \
            --gt-dir "$GT_DIR" \
            --csv "$RUN_PARENT/baseline_ce_3d_dice.csv" \
            | tee "$RUN_PARENT/baseline_ce_3d_dice.txt"
    else
        echo "Baseline predictions not found at $BASELINE_PRED; skipping post-hoc baseline evaluation."
    fi
fi

echo "Finished: $(date)"
