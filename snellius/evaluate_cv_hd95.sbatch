#!/usr/bin/env bash
#SBATCH --job-name=segthor_cv_hd95
#SBATCH --account=gpuuva084
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Patient-level 3D Dice and HD95 (mm) of the best-epoch validation predictions of every cross-validation run
# compared in evaluation/compare_cv_runs.py, one evaluation per run and fold, in parallel.
# Writes <prediction set>_3d_metrics.csv next to each run's existing Dice CSV; nothing is overwritten.

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
CV_DIR="$PROJECT/data/SEGTHOR_FULL_PREPROCESSED_split_cv"
SPACING="$PROJECT/data/SEGTHOR_FULL_PREPROCESSED/spacing.pkl"  # Effective spacing of the 384x384 slices

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1  # One evaluation per CPU

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"

# run directory (with fold_N inside) | prediction folder inside fold_N | output CSV name inside fold_N
RUNS=(
    "results/segthor_full_preprocessed_cv/cv/ce_dice_25d|best_epoch/val|best_epoch_3d_metrics.csv"
    "results/segthor_full_preprocessed_cv/cv/ce_dice_25d_aug|best_epoch/val|best_epoch_3d_metrics.csv"
    "results/segthor_full_preprocessed_cv/cv/weighted_ce_25d|best_epoch/val|best_epoch_3d_metrics.csv"
    "results/segthor_full_preprocessed_cv/cv_e50/weighted_ce_25d|best_epoch/val|best_epoch_3d_metrics.csv"
    "results/segthor_swin_unetr/cv|eval_slices|eval_slices_3d_metrics.csv"
)

tasks=()
for run in "${RUNS[@]}"; do
    IFS='|' read -r run_dir pred_subdir csv_name <<< "$run"
    for fold in 0 1 2 3 4; do
        if [[ -e "$run_dir/fold_$fold/$csv_name" ]]; then
            echo "Already exists; refusing to overwrite: $run_dir/fold_$fold/$csv_name" >&2
            exit 1
        fi
        tasks+=("$run_dir/fold_$fold|$pred_subdir|$csv_name|$fold")
    done
done
echo "Evaluations: ${#tasks[@]}"

evaluate() {
    IFS='|' read -r fold_dir pred_subdir csv_name fold <<< "$1"
    python -m evaluation.evaluate_saved_predictions \
        --pred-dir "$fold_dir/$pred_subdir" \
        --gt-dir "$CV_DIR/fold_$fold/val/gt" \
        --spacing-file "$SPACING" \
        --csv "$fold_dir/$csv_name" \
        > "$fold_dir/${csv_name%.csv}.txt" 2>&1 \
        && echo "done: $fold_dir" || { echo "FAILED: $fold_dir (see ${csv_name%.csv}.txt)"; return 1; }
}
export -f evaluate
export CV_DIR SPACING

printf '%s\n' "${tasks[@]}" | xargs -P "$SLURM_CPUS_PER_TASK" -I {} bash -c 'evaluate "$@"' _ {}

echo "Finished: $(date)"
