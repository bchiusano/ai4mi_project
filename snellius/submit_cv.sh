#!/usr/bin/env bash

# Submits every setting on every cross-validation fold (snellius/train_cv.sbatch):
#   baseline: ce_2d            run 1: ce_25d
#   run 2:    ce_dice_2d       run 3: ce_dice_25d
#   run 4:    weighted_ce_25d
# Usage: snellius/submit_cv.sh [RUN_TAG] [SETTINGS] [FOLDS]
#   e.g. snellius/submit_cv.sh cv                      # everything (5 settings x 5 folds)
#        snellius/submit_cv.sh cv "ce_2d ce_25d" "0"   # a subset

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
RUN_TAG="${1:-cv}"
SETTINGS="${2:-ce_2d ce_25d ce_dice_2d ce_dice_25d weighted_ce_25d}"
FOLDS="${3:-0 1 2 3 4}"

cd "$PROJECT"
mkdir -p logs

job_file="logs/cv_${RUN_TAG}.jobs"
: > "$job_file"

for setting in $SETTINGS; do
    for fold in $FOLDS; do
        name="segthor_cv_${setting}_f${fold}"
        job=$(sbatch --parsable \
            --job-name="$name" \
            --export=ALL,SETTING="$setting",FOLD="$fold",RUN_TAG="$RUN_TAG",WANDB_RUN_NAME="$setting-fold$fold-$RUN_TAG",WANDB_RUN_GROUP="$setting-$RUN_TAG" \
            snellius/train_cv.sbatch)
        echo "$setting fold $fold: $job" | tee -a "$job_file"
    done
done

echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -u $USER"
