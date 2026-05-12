#!/bin/bash
#SBATCH --job-name=werner_sq
#SBATCH --output=logs/werner_%A_%a.out
#SBATCH --error=logs/werner_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-63

# ---- Usage -----------------------------------------------------------------
# 1) On the login node, generate the parameter grid (creates params.txt):
#       python make_params.py --n-alpha 21 --NE-list 3 6 9
#    Then set --array=1-<number of lines in params.txt> in this script.
# 2) Submit:   sbatch run_werner.sbatch
# 3) When all tasks finish, collect:
#       python collect.py --in-dir results --out results.csv
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs results

PARAMS_FILE=${PARAMS_FILE:-params.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")
read -r ALPHA NE SEED <<< "$LINE"

echo "Task ${SLURM_ARRAY_TASK_ID}: alpha=${ALPHA}  NE=${NE}  seed=${SEED}"

# Keep BLAS single-threaded so cpus-per-task=1 is honored.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_werner.py \
    --alpha "${ALPHA}" \
    --NE "${NE}" \
    --seed "${SEED}" \
    --n-restarts 8 \
    --max-iter 800 \
    --out-dir results \
    --task-id "${SLURM_ARRAY_TASK_ID}"
