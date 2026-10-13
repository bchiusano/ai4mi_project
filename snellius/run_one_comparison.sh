#!/bin/bash
#SBATCH --job-name=compare_preds
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out

source ~/.bashrc          
source ai4mi/bin/activate
cd $SLURM_SUBMIT_DIR
python postprocessing/compare_saved_predictions.py --patient Patient_03