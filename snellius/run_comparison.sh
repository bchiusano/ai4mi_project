#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.out

source ~/.bashrc          
source ai4mi/bin/activate
cd $SLURM_SUBMIT_DIR
python postprocessing/comparison.py