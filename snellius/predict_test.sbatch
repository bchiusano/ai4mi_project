#!/usr/bin/env bash
#SBATCH --job-name=segthor_predict_test
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Segments the unlabelled SegTHOR test set (data/test.zip, Patient_41-60) with every cross-validated model and
# ensembles each model's 5 fold models by majority vote. Output: results/test_predictions/<model>/Patient_XX.nii.gz,
# labels 0-4 in the original CT space. The per-fold predictions stay on scratch (too many files for home).

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
BUNDLE="$PROJECT/models/segmentation_template_monai"
TEST_CT_DIR="$SCRATCH_PROJECT/data/segthor_test/test"
TEST_SLICES="$PROJECT/data/SEGTHOR_TEST_PREPROCESSED"
FOLD_PREDICTIONS="$SCRATCH_PROJECT/results/test_predictions"
OUTPUT="$PROJECT/results/test_predictions"

# name | results folder with fold_0..fold_4 | kind
MODELS=(
    "ce_dice_25d|$PROJECT/results/segthor_full_preprocessed_cv/cv/ce_dice_25d|25d"
    "ce_dice_25d_aug|$PROJECT/results/segthor_full_preprocessed_cv/cv/ce_dice_25d_aug|25d"
    "weighted_ce_25d|$PROJECT/results/segthor_full_preprocessed_cv/cv_e50/weighted_ce_25d|25d"
    "swin_unetr|$PROJECT/results/segthor_swin_unetr/cv|swin"
    "swin_unetr_eso_sampling|$PROJECT/results/segthor_swin_unetr/cv_esophagus_sampling|swin"
)

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="$BUNDLE"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch CUDA:", torch.cuda.get_device_name(0))'

for d in "$OUTPUT" "$FOLD_PREDICTIONS"; do
    if [[ -e "$d" ]]; then
        echo "Already exists; refusing to overwrite: $d" >&2
        exit 1
    fi
done
for model in "${MODELS[@]}"; do
    IFS='|' read -r name run_dir kind <<< "$model"
    for fold in 0 1 2 3 4; do
        weights="$run_dir/fold_$fold/$([[ $kind == 25d ]] && echo bestweights.pt || echo model.pt)"
        [[ -f "$weights" ]] || { echo "Missing weights: $weights" >&2; exit 1; }
    done
done

# Test CT volumes, unzipped once on scratch (also needed later for the CT headers of the outputs)
if [[ ! -d "$TEST_CT_DIR" ]]; then
    mkdir -p "$(dirname "$TEST_CT_DIR")"
    unzip -q "$PROJECT/data/test.zip" -d "$(dirname "$TEST_CT_DIR")"
fi
echo "Test volumes: $(ls "$TEST_CT_DIR"/Patient_*.nii.gz | wc -l)"

# 2D slices for the 2.5D models, prepared exactly like the training data
if [[ ! -f "$TEST_SLICES/preprocessing.json" ]]; then
    python -m inference.prepare_test_slices \
        --ct-dir "$TEST_CT_DIR" \
        --training-preprocessing-json data/SEGTHOR_FULL_PREPROCESSED/preprocessing.json \
        --dest "$TEST_SLICES"
fi

for model in "${MODELS[@]}"; do
    IFS='|' read -r name run_dir kind <<< "$model"
    echo "=== $name ($kind) from $run_dir"
    fold_dirs=()
    for fold in 0 1 2 3 4; do
        fold_dir="$FOLD_PREDICTIONS/$name/fold_$fold"
        fold_dirs+=("$fold_dir")
        if [[ $kind == 25d ]]; then
            python -m inference.predict_segthor_from_weights \
                --weights "$run_dir/fold_$fold/bestweights.pt" \
                --image-dir "$TEST_SLICES/img" \
                --output-dir "$fold_dir" \
                --context 1 --model enet --batch-size 32 --gpu
        else
            python -m monai.bundle run \
                --meta_file "$BUNDLE/configs/metadata.json" \
                --logging_file "$BUNDLE/configs/logging.conf" \
                --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/inference.yaml']" \
                --bundle_root "$BUNDLE" \
                --ckpt_path "$run_dir/fold_$fold/model.pt" \
                --dataset_dir "$TEST_CT_DIR" \
                --output_dir "$fold_dir" \
                --num_workers 8
        fi
    done
    python -m inference.ensemble_predictions \
        --kind "$([[ $kind == 25d ]] && echo slices || echo nifti)" \
        --fold-dirs "${fold_dirs[@]}" \
        --preprocessing-json "$TEST_SLICES/preprocessing.json" \
        --output-dir "$OUTPUT/$name"
    echo "Ensemble of $name: $(ls "$OUTPUT/$name"/*.nii.gz | wc -l) volumes"
done

echo "Finished: $(date)"
echo "Ensembled test predictions: $OUTPUT"
