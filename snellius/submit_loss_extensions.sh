#!/usr/bin/env bash

set -euo pipefail

PROJECT="$HOME/projects/ai4mi_project"
RUN_TAG="${1:-preliminary_3d_dice}"

cd "$PROJECT"
mkdir -p logs

dice_job=$(sbatch --parsable \
    --job-name=segthor_dice \
    --export=ALL,LOSS=dice,RUN_TAG="$RUN_TAG" \
    snellius/train_loss_extension.sbatch)

ce_dice_job=$(sbatch --parsable \
    --job-name=segthor_ce_dice \
    --export=ALL,LOSS=ce_dice,RUN_TAG="$RUN_TAG" \
    snellius/train_loss_extension.sbatch)

job_file="logs/loss_extensions_${RUN_TAG}.jobs"
printf 'DICE_JOB=%q\nCE_DICE_JOB=%q\n' "$dice_job" "$ce_dice_job" > "$job_file"

echo "Submitted both jobs independently."
echo "Dice job: $dice_job"
echo "CE+Dice job: $ce_dice_job"
echo "Saved job IDs to: $job_file"
echo "Monitor with: squeue -j $dice_job,$ce_dice_job"
echo "Dice log: logs/segthor_dice-$dice_job.out"
echo "CE+Dice log: logs/segthor_ce_dice-$ce_dice_job.out"
