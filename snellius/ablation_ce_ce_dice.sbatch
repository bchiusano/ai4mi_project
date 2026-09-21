#!/usr/bin/env bash
#SBATCH --job-name=segthor_ablation
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=logs/segthor_ablation-%j.out
#SBATCH --error=logs/segthor_ablation-%j.err

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
SOURCE="$PROJECT/data/segthor_part1"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project/ablation_ce_ce_dice_v4"

CLEAN_DATA="$SCRATCH_PROJECT/data/segthor_part1_clean"
SLICED_DATA="$SCRATCH_PROJECT/data/SEGTHOR_CLEAN"

RESULT_ROOT="$SCRATCH_PROJECT/results"
CE_RESULT="$RESULT_ROOT/ce"
CE_DICE_RESULT="$RESULT_ROOT/ce_dice"

cd "$PROJECT"

module purge
module load 2025
source ai4mi/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

export WANDB_PROJECT="ai4mi-segthor"
export WANDB_RUN_GROUP="ce-vs-ce-dice"
export WANDB_LOG_MODEL="false"

echo "Started: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -d "$SOURCE/train" ]]; then
    echo "Raw SegTHOR source is missing: $SOURCE/train" >&2
    exit 1
fi

if [[ -e "$CLEAN_DATA" || -e "$SLICED_DATA" || -e "$CE_RESULT" || -e "$CE_DICE_RESULT" ]]; then
    echo "An ablation output already exists; refusing to overwrite it." >&2
    echo "CLEAN_DATA=$CLEAN_DATA" >&2
    echo "SLICED_DATA=$SLICED_DATA" >&2
    echo "CE_RESULT=$CE_RESULT" >&2
    echo "CE_DICE_RESULT=$CE_DICE_RESULT" >&2
    exit 1
fi

if [[ -e data/SEGTHOR_CORRECTED && ! -L data/SEGTHOR_CORRECTED ]]; then
    echo "data/SEGTHOR_CORRECTED exists but is not a symlink; refusing to replace it." >&2
    exit 1
fi

mkdir -p "$SCRATCH_PROJECT/data" "$RESULT_ROOT"

echo "Preparing corrected labels..."
python split_merged_label_cc.py \
    --source-root "$SOURCE" \
    --dest-root "$CLEAN_DATA"

echo "Slicing corrected data..."
python slice_segthor.py \
    --source_dir "$CLEAN_DATA" \
    --dest_dir "$SLICED_DATA" \
    --shape 256 256 \
    --retains 5 \
    --seed 0 \
    --fold 0 \
    --process 8

train_count=$(find "$SLICED_DATA/train/img" -type f -name '*.png' | wc -l)
val_count=$(find "$SLICED_DATA/val/img" -type f -name '*.png' | wc -l)
if [[ "$train_count" -ne 2817 || "$val_count" -ne 915 ]]; then
    echo "Unexpected slice counts: train=$train_count val=$val_count" >&2
    exit 1
fi

ln -sfn "$SLICED_DATA" data/SEGTHOR_CORRECTED
echo "SEGTHOR_CORRECTED -> $(readlink -f data/SEGTHOR_CORRECTED)"


echo "Running cross-entropy baseline..."
export WANDB_RUN_NAME="corrected-ce"
python -u main.py \
    --epochs 25 \
    --dataset SEGTHOR_CORRECTED \
    --mode full \
    --loss ce \
    --dest "$CE_RESULT" \
    --gpu

echo "Running cross-entropy plus Dice ablation..."
export WANDB_RUN_NAME="corrected-ce-dice"
python -u main.py \
    --epochs 25 \
    --dataset SEGTHOR_CORRECTED \
    --mode full \
    --loss ce_dice \
    --dest "$CE_DICE_RESULT" \
    --gpu

echo "Finished: $(date)"
echo "CE results:      $CE_RESULT"
echo "CE+Dice results: $CE_DICE_RESULT"