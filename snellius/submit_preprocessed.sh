#!/usr/bin/env bash

# Submits settings on the fixed train/val/test split of SEGTHOR_FULL_PREPROCESSED
# (one job per setting, snellius/train_cv.sbatch). The best model of each run is
# also evaluated on the 6 held-out test patients.
# The split folders are built once with:
#   python -m preprocessing.make_split_dirs --source data/SEGTHOR_FULL_PREPROCESSED \
#       --split data/SEGTHOR_FULL_PREPROCESSED/split.json --dest data/SEGTHOR_FULL_PREPROCESSED_split
# Usage: snellius/submit_preprocessed.sh [RUN_TAG] [SETTINGS]
#   e.g. snellius/submit_preprocessed.sh split "ce_dice_2d ce_dice_25d"

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
RUN_TAG="${1:-split}"
SETTINGS="${2:-ce_2d ce_25d ce_dice_2d ce_dice_25d weighted_ce_25d}"
DATA_DIR="$PROJECT/data/SEGTHOR_FULL_PREPROCESSED_split"

cd "$PROJECT"
mkdir -p logs

if [[ ! -d "$DATA_DIR/train/img" ]]; then
    echo "Split folders missing: $DATA_DIR (see the top of this script to build them)" >&2
    exit 1
fi

job_file="logs/preprocessed_${RUN_TAG}.jobs"
: > "$job_file"

for setting in $SETTINGS; do
    name="segthor_pre_${setting}"
    job=$(sbatch --parsable \
        --job-name="$name" \
        --export=ALL,SETTING="$setting",DATA_DIR="$DATA_DIR",RESULTS_NAME=segthor_full_preprocessed,RUN_SUBDIR=split,RUN_TAG="$RUN_TAG",WANDB_RUN_NAME="pre-$setting-$RUN_TAG",WANDB_RUN_GROUP="preprocessed-$RUN_TAG" \
        snellius/train_cv.sbatch)
    echo "$setting: $job" | tee -a "$job_file"
done

echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -u $USER"
