#!/usr/bin/env bash
#SBATCH --job-name=slice_split_segthor_full
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Prepares the full 40-patient SegTHOR training set for cross-validation:
#   1. unzips data/segthor_train_full.zip to scratch,
#   2. slices all patients to 256x256 PNGs (slice_segthor.py, no validation split),
#   3. builds patient-level folds as symlink folders (make_cv_folds.py).
# The labels in this archive are the original ones, so no label repair is applied.
# Usage: sbatch snellius/slice_split_segthor_full.sbatch   (FOLDS and SEED can be overridden)

set -euo pipefail

FOLDS="${FOLDS:-5}"
SEED="${SEED:-0}"

PROJECT="$HOME/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
ZIP="$PROJECT/data/segthor_train_full.zip"
RAW_DIR="$SCRATCH_PROJECT/data/segthor_train_full"
SLICED_DIR="$SCRATCH_PROJECT/data/SEGTHOR_FULL_sliced"
CV_DIR="$SCRATCH_PROJECT/data/SEGTHOR_FULL_CV"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Started: $(date)"
echo "Git commit: $(git rev-parse HEAD)"

for d in "$SLICED_DIR" "$CV_DIR"; do
    if [[ -e "$d" ]]; then
        echo "Already exists; refusing to overwrite: $d" >&2
        exit 1
    fi
done

if [[ ! -d "$RAW_DIR/train" ]]; then
    mkdir -p "$RAW_DIR"
    unzip -q "$ZIP" -d "$RAW_DIR"
fi
n_patients=$(find "$RAW_DIR/train" -mindepth 1 -maxdepth 1 -type d -name 'Patient_*' | wc -l)
echo "Patients in archive: $n_patients"
if [[ "$n_patients" -ne 40 ]]; then
    echo "Expected 40 patients in $RAW_DIR/train" >&2
    exit 1
fi

# --retains 0: every patient goes to $SLICED_DIR/train, the folds are made below
python -m preprocessing.slice_segthor \
    --source_dir "$RAW_DIR" \
    --dest_dir "$SLICED_DIR" \
    --retains 0 \
    --process "$SLURM_CPUS_PER_TASK"

python preprocessing/make_cv_folds.py \
    --source "$SLICED_DIR/train" \
    --dest "$CV_DIR" \
    --folds "$FOLDS" \
    --seed "$SEED" \
    | tee "$SCRATCH_PROJECT/data/SEGTHOR_FULL_CV_folds.txt"

cp "$CV_DIR/folds.json" "$PROJECT/preprocessing/segthor_full_cv_folds.json"
echo "Fold assignment copied to preprocessing/segthor_full_cv_folds.json (commit it to record the split)"
echo "Finished: $(date)"
