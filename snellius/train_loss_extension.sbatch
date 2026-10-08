#!/usr/bin/env bash
#SBATCH --job-name=segthor_ext
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

#: "${LOSS:?Submit with LOSS=dice or LOSS=ce_dice}"
LOSS="${LOSS:-dice}"
RUN_TAG="${RUN_TAG:-preliminary_3d_dice}"

case "$LOSS" in
    dice|ce_dice) ;;
    *)
        echo "Unsupported LOSS=$LOSS; expected dice or ce_dice" >&2
        exit 2
        ;;
esac

PROJECT="$HOME/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
EXPECTED_DATA="$SCRATCH_PROJECT/data/SEGTHOR"
DATA_LINK="$PROJECT/data/SEGTHOR"
RUN_DIR="$SCRATCH_PROJECT/results/segthor/$RUN_TAG/$LOSS"
RUN_PARENT="$SCRATCH_PROJECT/results/segthor/$RUN_TAG"
GT_DIR="$DATA_LINK/val/gt"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source ai4mi/bin/activate

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
python -m py_compile main.py utils.py losses.py \
    data_loading/dataset.py models/ENet.py models/ShallowNet.py \
    plotting/plot.py evaluation/summarize_segthor_run.py \
    evaluation/evaluate_saved_predictions.py
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
if [[ "$train_count" -ne 6432 || "$val_count" -ne 988 ]]; then
    echo "Unexpected slice counts: train=$train_count val=$val_count" >&2
    exit 1
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

# --- wandb ---
export WANDB_PROJECT="ai4mi-segthor"
export WANDB_ENTITY="b-chiusano-uva"        # must match your teammate's project
export WANDB_RUN_GROUP="$RUN_TAG"
export WANDB_RUN_NAME="segthor-$LOSS-$RUN_TAG"
export WANDB_DIR="$RUN_DIR"                        # keep wandb files on scratch, not in $HOME
export WANDB_MODE="${WANDB_MODE:-online}"          # submit with WANDB_MODE=offline if needed
# export WANDB_API_KEY="$(cat "$HOME/.wandb_key")" # only if you didn't run `wandb login`

python -c 'import wandb; print("wandb", wandb.__version__)'

python -u main.py \
    --epochs 25 \
    --dataset SEGTHOR \
    --mode full \
    --loss "$LOSS" \
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
    --gt-dir "$GT_DIR" \
    --csv "$RUN_DIR/best_epoch_3d_dice.csv" \
    | tee "$RUN_DIR/best_epoch_3d_dice.txt"

if [[ "$LOSS" == "dice" ]]; then
    BASELINE_PRED="$SCRATCH_PROJECT/results/segthor/baseline/best_epoch/val"
    if [[ -d "$BASELINE_PRED" ]]; then
        python -m evaluation.evaluate_saved_predictions \
            --pred-dir "$BASELINE_PRED" \
            --gt-dir "$GT_DIR" \
            --csv "$RUN_PARENT/baseline_ce_3d_dice.csv" \
            | tee "$RUN_PARENT/baseline_ce_3d_dice.txt"
    else
        echo "Baseline predictions not found at $BASELINE_PRED; skipping post-hoc baseline evaluation."
    fi
fi

echo "Finished: $(date)"
