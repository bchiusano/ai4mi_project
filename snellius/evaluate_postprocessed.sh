#!/bin/bash
#SBATCH --job-name=eval_postproc
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out

source ~/.bashrc          
source ai4mi/bin/activate
cd $SLURM_SUBMIT_DIR

#for m in ce_dice_25d ce_dice_25d_aug swin_unetr swin_unetr_eso_sampling weighted_ce_25d; do
#  echo "=== $m ==="
# python -m evaluation.evaluate_saved_predictions --pred-dir results/heldout_predictions_postprocessed_slices/$m --gt-dir data/SEGTHOR_FULL/gt --split-file data/SEGTHOR_FULL/heldout_split.json --subset test --spacing-file data/SEGTHOR_FULL/spacing.pkl --in-plane-spacing-scale 1 --csv results/heldout_predictions/metrics_postprocessed/$m.csv
#done

for m in ce_dice_25d ce_dice_25d_aug weighted_ce_25d; do
  echo "=== $m ==="
  python -m evaluation.evaluate_saved_predictions --pred-dir results/heldout_predictions_postprocessed_pngs/$m --gt-dir data/SEGTHOR_FULL/gt --split-file data/SEGTHOR_FULL/heldout_split.json --subset test --spacing-file data/SEGTHOR_FULL/spacing.pkl --in-plane-spacing-scale 1 --csv results/heldout_predictions/metrics_postprocessed_png/$m.csv
done