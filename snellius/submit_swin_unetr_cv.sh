#!/usr/bin/env bash

# Submits SwinUNETR fine-tuning on every cross-validation fold (snellius/train_swin_unetr.sbatch).
# Usage: snellius/submit_swin_unetr_cv.sh [RUN_TAG] [FOLDS]
#   e.g. snellius/submit_swin_unetr_cv.sh cv          # all 5 folds
#        snellius/submit_swin_unetr_cv.sh cv "0 1"    # a subset

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
RUN_TAG="${1:-cv}"
FOLDS="${2:-0 1 2 3 4}"

cd "$PROJECT"
mkdir -p logs

job_file="logs/swin_unetr_${RUN_TAG}.jobs"
: > "$job_file"

for fold in $FOLDS; do
    name="segthor_swin_unetr_f${fold}"
    job=$(sbatch --parsable \
        --job-name="$name" \
        --export=ALL,FOLD="$fold",RUN_TAG="$RUN_TAG" \
        snellius/train_swin_unetr.sbatch)
    echo "swin_unetr fold $fold: $job" | tee -a "$job_file"
done

echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -u $USER"
