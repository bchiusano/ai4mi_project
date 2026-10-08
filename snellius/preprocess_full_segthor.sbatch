#!/usr/bin/env bash
#SBATCH --job-name=segthor_preprocess
#SBATCH --account=gpuuva084
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT="${PROJECT:-$HOME/projects/ai4mi_project}"
SCRATCH_PROJECT="${SCRATCH_PROJECT:-/scratch-shared/$USER/ai4mi_project}"
SOURCE_ZIP="$SCRATCH_PROJECT/source/segthor_train_full.zip"
RAW_DATA="$SCRATCH_PROJECT/data/segthor_train_full"
FINAL_DATA="$SCRATCH_PROJECT/data/SEGTHOR_FULL"
QC_DIR="$SCRATCH_PROJECT/results/preprocessing/final"
BUILD_ROOT="$SCRATCH_PROJECT/data/.preprocess-build-${SLURM_JOB_ID}"

cd "$PROJECT"
mkdir -p logs "$QC_DIR"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
python --version

python -m unittest discover -s tests -v

patient_count() {
    local root="$1"
    if [[ ! -d "$root/train" ]]; then
        printf '0\n'
        return
    fi
    find "$root/train" -mindepth 1 -maxdepth 1 -type d -name 'Patient_*' | wc -l
}

if [[ "$(patient_count "$RAW_DATA")" -eq 40 ]]; then
    echo "Reusing extracted source: $RAW_DATA"
else
    if [[ -e "$RAW_DATA" ]]; then
        echo "Incomplete source directory: $RAW_DATA" >&2
        exit 1
    fi
    if [[ ! -f "$SOURCE_ZIP" ]]; then
        echo "Missing source archive: $SOURCE_ZIP" >&2
        exit 1
    fi

    expected_hash=$(awk 'NR == 1 {print $1}' data/segthor_train_full.sha256)
    actual_hash=$(sha256sum "$SOURCE_ZIP" | awk '{print $1}')
    if [[ "$actual_hash" != "$expected_hash" ]]; then
        echo "Archive checksum mismatch." >&2
        echo "Expected: $expected_hash" >&2
        echo "Actual:   $actual_hash" >&2
        exit 1
    fi

    mkdir -p "$BUILD_ROOT/raw"
    unzip -q "$SOURCE_ZIP" -d "$BUILD_ROOT/raw"
    if [[ "$(patient_count "$BUILD_ROOT/raw")" -ne 40 ]]; then
        echo "The archive does not contain exactly 40 patients." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$RAW_DATA")"
    mv "$BUILD_ROOT/raw" "$RAW_DATA"
fi

if [[ -d "$FINAL_DATA" ]]; then
    echo "Final dataset already exists; validating without overwriting it."
else
    mkdir -p "$BUILD_ROOT"
    build="$BUILD_ROOT/SEGTHOR_FULL"
    python -u -m preprocessing.slice_segthor \
        --source_dir "$RAW_DATA" \
        --dest_dir "$build" \
        --shape 384 384 \
        --target-spacing 1.0 1.0 \
        --window -1000 1000 \
        --validation-count 6 \
        --test-count 6 \
        --test-pool-start 21 \
        --split-seed 0 \
        --process "$SLURM_CPUS_PER_TASK"
    python -u -m preprocessing.validate_processed_segthor "$build"
    mv "$build" "$FINAL_DATA"
fi

python -u -m preprocessing.validate_processed_segthor "$FINAL_DATA" \
    | tee "$QC_DIR/validation.json"
cp "$FINAL_DATA/split.json" "$QC_DIR/split.json"

mkdir -p "$PROJECT/data"
link="$PROJECT/data/SEGTHOR_FULL"
if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$FINAL_DATA")" ]] \
        || { echo "Existing link points to the wrong dataset: $link" >&2; exit 1; }
elif [[ -e "$link" ]]; then
    echo "$link exists and is not a symlink; refusing to replace it." >&2
    exit 1
else
    ln -s "$FINAL_DATA" "$link"
fi

echo "Final dataset: $FINAL_DATA"
echo "Fixed split:"
cat "$FINAL_DATA/split.json"
echo "Finished: $(date)"
