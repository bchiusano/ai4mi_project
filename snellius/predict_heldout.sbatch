#!/usr/bin/env bash
#SBATCH --job-name=segthor_predict_heldout
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Segments the 8 held-out patients (data/SEGTHOR_FULL_PREPROCESSED_split/val, in none of the 5 CV folds) with the
# 5 fold models of every cross-validated model, ensembles them by majority vote, and scores the ensembles.
# Output in results/heldout_predictions/:
#   <2.5D model>/Patient_XX_ZZZZ.png  voted slices, raw (384x384, labels x 63, same grid as the GT slices)
#   <SwinUNETR model>/Patient_XX.nii.gz  voted volumes in the original CT space (the model's native output)
#   metrics/<model>.csv / .txt  patient-level 3D Dice and HD95 (mm) against the GT slices
# SwinUNETR predictions are put on the slice grid for scoring only (on scratch), like in the CV evaluation.

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
BUNDLE="$PROJECT/models/segmentation_template_monai"
HELDOUT="$PROJECT/data/SEGTHOR_FULL_PREPROCESSED_split/val"  # img/ and gt/ slices of the 8 patients
HELDOUT_CT_DIR="$SCRATCH_PROJECT/data/segthor_heldout"        # their original CT volumes, for SwinUNETR
FOLD_PREDICTIONS="$SCRATCH_PROJECT/results/heldout_predictions"
OUTPUT="$PROJECT/results/heldout_predictions"
SPACING="$PROJECT/data/SEGTHOR_FULL_PREPROCESSED/spacing.pkl"

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

patients=$(ls "$HELDOUT/img" | grep -o '^Patient_[0-9]*' | sort -u)
echo "Held-out patients: $(echo $patients)"
# The held-out patients must not be in any cross-validation fold
python - "$patients" <<'EOF'
import json, sys
held = set(sys.argv[1].split())
folds = json.load(open("data/SEGTHOR_FULL_PREPROCESSED_split_cv/folds.json"))["folds"]
overlap = held & set().union(*[set(f["train"]) | set(f["val"]) for f in folds])
assert len(held) == 8 and not overlap, f"held-out patients {sorted(held)} overlap the CV folds: {sorted(overlap)}"
print("Held-out patients are in none of the CV folds")
EOF

if [[ ! -d "$HELDOUT_CT_DIR" ]]; then
    mkdir -p "$HELDOUT_CT_DIR"
    for p in $patients; do
        unzip -q -p "$PROJECT/data/segthor_train_full.zip" "train/$p/$p.nii.gz" > "$HELDOUT_CT_DIR/$p.nii.gz"
    done
fi
echo "Held-out CT volumes: $(ls "$HELDOUT_CT_DIR"/Patient_*.nii.gz | wc -l)"

mkdir -p "$OUTPUT/metrics"
score_jobs=()
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
                --image-dir "$HELDOUT/img" \
                --output-dir "$fold_dir" \
                --context 1 --model enet --batch-size 32 --gpu
        else
            python -m monai.bundle run \
                --meta_file "$BUNDLE/configs/metadata.json" \
                --logging_file "$BUNDLE/configs/logging.conf" \
                --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/inference.yaml']" \
                --bundle_root "$BUNDLE" \
                --ckpt_path "$run_dir/fold_$fold/model.pt" \
                --dataset_dir "$HELDOUT_CT_DIR" \
                --output_dir "$fold_dir" \
                --num_workers 8
        fi
    done

    if [[ $kind == 25d ]]; then
        python -m inference.ensemble_predictions --kind slices --fold-dirs "${fold_dirs[@]}" --output-dir "$OUTPUT/$name"
        scored="$OUTPUT/$name"
    else
        python -m inference.ensemble_predictions --kind nifti --fold-dirs "${fold_dirs[@]}" --output-dir "$OUTPUT/$name"
        scored="$FOLD_PREDICTIONS/$name/ensemble_slices"
        python -m evaluation.nifti_predictions_to_slices \
            --pred-dir "$OUTPUT/$name" \
            --preprocessing-json data/SEGTHOR_FULL_PREPROCESSED/preprocessing.json \
            --output-dir "$scored"
    fi

    # Dice and HD95 on the CPUs while the next model runs on the GPU
    OMP_NUM_THREADS=2 python -m evaluation.evaluate_saved_predictions \
        --pred-dir "$scored" \
        --gt-dir "$HELDOUT/gt" \
        --spacing-file "$SPACING" \
        --csv "$OUTPUT/metrics/$name.csv" \
        > "$OUTPUT/metrics/$name.txt" 2>&1 &
    score_jobs+=($!)
done

for job in "${score_jobs[@]}"; do
    wait "$job"
done
for model in "${MODELS[@]}"; do
    IFS='|' read -r name _ _ <<< "$model"
    echo "--- $name"
    grep -E "Mean foreground" "$OUTPUT/metrics/$name.txt"
done

echo "Finished: $(date)"
