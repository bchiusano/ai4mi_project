#!/usr/bin/env bash
#SBATCH --job-name=unet_smoke
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=logs/unet_smoke-%j.out
#SBATCH --error=logs/unet_smoke-%j.err

set -euo pipefail

PROJECT="$HOME/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
EXPECTED_DATA="$SCRATCH_PROJECT/data/SEGTHOR_CLEAN_v2"
DATA_LINK="$PROJECT/data/SEGTHOR_CLEAN"
RUN_DIR="$SCRATCH_PROJECT/results/segthor_clean/unet_smoke_${SLURM_JOB_ID}"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source ai4mi/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_MODE=offline
export WANDB_PROJECT=ai4mi-segthor
export WANDB_RUN_NAME="unet-smoke-${SLURM_JOB_ID}"

echo "Git commit: $(git rev-parse HEAD)"
python --version

python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'

python -m py_compile main.py models/UNet.py

python -c 'import torch; from models.UNet import UNet; model = UNet(1, 5); output = model(torch.randn(2, 1, 256, 256)); assert output.shape == (2, 5, 256, 256); print("U-Net output:", output.shape)'

if [[ ! -d "$EXPECTED_DATA/train/img" || ! -d "$EXPECTED_DATA/val/img" ]]; then
    echo "Corrected dataset missing: $EXPECTED_DATA" >&2
    exit 1
fi

if [[ -e "$DATA_LINK" && ! -L "$DATA_LINK" ]]; then
    echo "$DATA_LINK exists but is not a symbolic link." >&2
    exit 1
fi

ln -sfn "$EXPECTED_DATA" "$DATA_LINK"
mkdir -p "$RUN_DIR"

python -u main.py \
    --epochs 1 \
    --dataset SEGTHOR_CLEAN \
    --model unet \
    --mode full \
    --loss ce \
    --dest "$RUN_DIR" \
    --gpu \
    --debug

echo "Smoke test completed successfully."
echo "Results: $RUN_DIR"