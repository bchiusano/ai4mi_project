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

python -m postprocessing.close_esophagus results/heldout_predictions/ce_dice_25d results/heldout_predictions_eso_close/ce_dice_25d_z5 5 3