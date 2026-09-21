#!/usr/bin/env bash
#SBATCH --job-name=ce_3d_eval
#SBATCH --account=gpuuva084
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ce_3d_eval-%j.out
#SBATCH --error=logs/ce_3d_eval-%j.err

set -euo pipefail

PROJECT="$HOME/projects/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
CE_RUN="$SCRATCH_PROJECT/results/segthor_clean/baseline_patient19_fixed"
GT_DIR="$SCRATCH_PROJECT/data/SEGTHOR_CLEAN_v2/val/gt"
OUTPUT_DIR="$SCRATCH_PROJECT/results/segthor_clean/preliminary_3d_dice_v1/ce_all_epochs"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "CE run: $CE_RUN"
echo "Ground truth: $GT_DIR"

epoch_count=$(find "$CE_RUN" -mindepth 1 -maxdepth 1 -type d -name 'iter[0-9][0-9][0-9]' | wc -l)
gt_count=$(find "$GT_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)
echo "Saved CE epoch folders: $epoch_count"
echo "Validation ground-truth slices: $gt_count"

if [[ "$epoch_count" -ne 25 ]]; then
    echo "Expected 25 saved CE epoch folders, found $epoch_count." >&2
    exit 1
fi
if [[ "$gt_count" -ne 915 ]]; then
    echo "Expected 915 validation ground-truth slices, found $gt_count." >&2
    exit 1
fi

python -u evaluate_saved_epochs.py \
    --run-dir "$CE_RUN" \
    --gt-dir "$GT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prefix baseline_ce \
    --workers "$SLURM_CPUS_PER_TASK"

echo "Finished: $(date)"
