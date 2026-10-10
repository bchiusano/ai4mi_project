#!/usr/bin/env bash

# Submits 5-fold cross-validation within the 32 training patients of the fixed
# train/val split of SEGTHOR_FULL_PREPROCESSED (one job per setting and fold,
# snellius/train_cv.sbatch). The 8 validation patients of the split are not used.
# The folds are built once with:
#   python preprocessing/make_cv_folds.py --source data/SEGTHOR_FULL_PREPROCESSED_split/train \
#       --dest data/SEGTHOR_FULL_PREPROCESSED_split_cv --folds 5 --seed 0
# Usage: snellius/submit_cv_preprocessed.sh [RUN_TAG] [SETTINGS] [FOLDS]
#   e.g. EPOCHS=35 TIME=06:00:00 snellius/submit_cv_preprocessed.sh cv "ce_dice_25d"

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
RUN_TAG="${1:-cv}"
SETTINGS="${2:-ce_dice_25d}"
FOLDS="${3:-0 1 2 3 4}"
TIME="${TIME:-08:00:00}"  # Slurm time limit per job
CV_DIR="$PROJECT/data/SEGTHOR_FULL_PREPROCESSED_split_cv"

cd "$PROJECT"
mkdir -p logs

if [[ ! -f "$CV_DIR/folds.json" ]]; then
    echo "Cross-validation folds missing: $CV_DIR (see the top of this script to build them)" >&2
    exit 1
fi

job_file="logs/cv_preprocessed_${RUN_TAG}.jobs"
: > "$job_file"

for setting in $SETTINGS; do
    for fold in $FOLDS; do
        job=$(sbatch --parsable \
            --time="$TIME" \
            --job-name="segthor_precv_${setting}_f${fold}" \
            --export=ALL,SETTING="$setting",DATA_DIR="$CV_DIR/fold_$fold",RESULTS_NAME=segthor_full_preprocessed_cv,RUN_SUBDIR="fold_$fold",RUN_TAG="$RUN_TAG",WANDB_RUN_NAME="precv-$setting-fold$fold-$RUN_TAG",WANDB_RUN_GROUP="precv-$setting-$RUN_TAG" \
            snellius/train_cv.sbatch)
        echo "$setting fold $fold: $job" | tee -a "$job_file"
    done
done

echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -u $USER"
