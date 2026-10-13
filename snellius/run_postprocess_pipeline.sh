#!/bin/bash
#SBATCH --job-name=postproc_final
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

# Usage:   sbatch run_pipeline.sh
# Variant: TAG=trachea_close CLOSE=1,3 sbatch run_pipeline.sh   (own output folders per TAG)

cd $SLURM_SUBMIT_DIR

#set -e

source ~/.bashrc          
source ai4mi/bin/activate

TAG=${TAG:-final}
LCC=${LCC:-2,3}        # classes that get largest-component (2 heart, 3 trachea)
BRIDGE=${BRIDGE:-1}    # classes whose gaps get bridged (1 esophagus)
CLOSE=${CLOSE:-1}      # classes that get z-closing (1 esophagus; add 3 for trachea)

PNG_MODELS="ce_dice_25d ce_dice_25d_aug weighted_ce_25d"     # predictions are PNG slices
NIFTI_MODELS="swin_unetr swin_unetr_eso_sampling"            # predictions are NIfTI

RAW=results/heldout_predictions
OUT=results/heldout_${TAG}
SLICES=results/heldout_${TAG}_slices
METRICS=results/heldout_${TAG}_metrics
COMPARISON=results/heldout_${TAG}_comparison
PREP=data/SEGTHOR_FULL/preprocessing.json
GT_NII=data/segthor_train_full/train

evaluate () {   # $1 = folder of PNG slices, $2 = model name
  python -m evaluation.evaluate_saved_predictions --pred-dir "$1" --gt-dir data/SEGTHOR_FULL/gt --split-file data/SEGTHOR_FULL/heldout_split.json --subset test --spacing-file data/SEGTHOR_FULL/spacing.pkl --in-plane-spacing-scale 1 --csv "$METRICS/$2.csv"
}

echo "=== 1. postprocess (TAG=$TAG LCC=$LCC BRIDGE=$BRIDGE CLOSE=$CLOSE) ==="
python -m postprocessing.postprocess_pipeline --input-dir $RAW --output-dir $OUT --models $PNG_MODELS $NIFTI_MODELS --lcc-classes "$LCC" --bridge-classes "$BRIDGE" --close-classes "$CLOSE"

echo "=== 2. NIfTI models: postprocessed NIfTI -> slice grid (needed for evaluation) ==="
for m in $NIFTI_MODELS; do
  python -m evaluation.nifti_predictions_to_slices --pred-dir $OUT/$m --preprocessing-json $PREP --output-dir $SLICES/$m
done

echo "=== 3. evaluate ==="
mkdir -p $METRICS
for m in $PNG_MODELS; do echo "--- $m"; evaluate $OUT/$m $m; done
for m in $NIFTI_MODELS; do echo "--- $m"; evaluate $SLICES/$m $m; done

echo "=== 4. PNG models: slices -> NIfTI for visualisation (raw and postprocessed) ==="
for m in $PNG_MODELS; do
  python -m postprocessing.png_slices_to_nifti $RAW/$m results/nifti_views/raw/$m $PREP $GT_NII
  python -m postprocessing.png_slices_to_nifti $OUT/$m results/nifti_views/$TAG/$m $PREP $GT_NII
done

echo "=== 5. compare before vs after ==="
python -m evaluation.compare_postprocessing --before-dir $RAW/metrics --after-dir $METRICS --output-dir $COMPARISON