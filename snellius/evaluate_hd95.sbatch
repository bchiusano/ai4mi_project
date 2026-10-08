#!/usr/bin/env bash
#SBATCH --job-name=segthor_hd95
#SBATCH --account=gpuuva084
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/segthor_hd95-%j.out
#SBATCH --error=logs/segthor_hd95-%j.err

set -euo pipefail

PROJECT="$HOME/projects/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
DATASET="$SCRATCH_PROJECT/data/SEGTHOR_CLEAN_v2"
RESULT_ROOT="$SCRATCH_PROJECT/results/segthor_clean/preliminary_3d_dice_v1"
OUTPUT_DIR="$RESULT_ROOT/hd95"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

mkdir -p logs "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Started: $(date)"
echo "Host: $(hostname)"

# This spacing file predates the corrected resize-aware spacing calculation.
# SEGTHOR_CLEAN_v2 was resized from 512x512 to 256x256, so x/y spacing doubles.
COMMON_ARGS=(
    --gt-dir "$DATASET/val/gt"
    --spacing-file "$DATASET/spacing.pkl"
    --in-plane-spacing-scale 2
)

python -u evaluate_saved_predictions.py \
    --pred-dir "$SCRATCH_PROJECT/results/segthor_clean/baseline_patient19_fixed/best_epoch/val" \
    "${COMMON_ARGS[@]}" \
    --csv "$OUTPUT_DIR/ce_metrics.csv" \
    | tee "$OUTPUT_DIR/ce_summary.txt"

python -u evaluate_saved_predictions.py \
    --pred-dir "$RESULT_ROOT/dice/best_epoch/val" \
    "${COMMON_ARGS[@]}" \
    --csv "$OUTPUT_DIR/dice_metrics.csv" \
    | tee "$OUTPUT_DIR/dice_summary.txt"

python -u evaluate_saved_predictions.py \
    --pred-dir "$RESULT_ROOT/ce_dice/best_epoch/val" \
    "${COMMON_ARGS[@]}" \
    --csv "$OUTPUT_DIR/ce_dice_metrics.csv" \
    | tee "$OUTPUT_DIR/ce_dice_summary.txt"

echo "Finished: $(date)"
echo "Outputs: $OUTPUT_DIR"
