#!/bin/bash
#SBATCH --job-name=esq_scan
#SBATCH --output=logs/esq_%A_%a.out
#SBATCH --error=logs/esq_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-55

# ---- Usage -----------------------------------------------------------------
# 1) Generate the parameter grid:
#       python make_params_esq.py
# 2) Submit:   sbatch run_esq.sbatch
# 3) Collect & plot:  python collect_and_plot_esq.py
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs results_esq

PARAMS_FILE=${PARAMS_FILE:-params_esq.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")
NE=$(echo "$LINE" | awk '{print $1}')
ALPHA=$(echo "$LINE" | awk '{print $2}')

echo "Task ${SLURM_ARRAY_TASK_ID}: NE=${NE}  alpha=${ALPHA}"

# 1. Initialize the module system (crucial for Sheldon)
source /etc/profile.d/modules.sh

# 2. Force JAX to use CPU (prevents errors if it tries to look for CUDA)
export JAX_PLATFORMS=cpu

# 3. Activate your virtual environment
# Ensure the path is correct for your scratch directory
source /home/noafeld/my_jax_env/bin/activate

# Keep BLAS single-threaded so cpus-per-task=1 is honored.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_werner.py \
    --NE "${NE}" \
    --alpha "${ALPHA}" \
    --n-restarts 256 \
    --max-iter 2000 \
    --seed 0 \
    --out-dir results_esq \
    --task-id "${SLURM_ARRAY_TASK_ID}"
