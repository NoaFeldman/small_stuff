#!/bin/bash
#SBATCH --job-name=werner_sq
#SBATCH --output=logs/werner_%A_%a.out
#SBATCH --error=logs/werner_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-3

# ---- Usage -----------------------------------------------------------------
# 1) Generate the parameter grid (one NE per line):
#       python make_params.py --NE-list 3 6 9
#    Update --array=1-<number of lines> if you change --NE-list.
# 2) Submit:   sbatch run_werner.sbatch
# 3) Collect:  python collect.py --in-dir results --out results.csv
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs results

PARAMS_FILE=${PARAMS_FILE:-params.txt}
NE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")

echo "Task ${SLURM_ARRAY_TASK_ID}: NE=${NE}"

# Keep BLAS single-threaded so cpus-per-task=1 is honored.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu

python run_werner.py \
    --NE "${NE}" \
    --alpha-min -1.0 \
    --alpha-max  1.0 \
    --n-alpha 21 \
    --n-restarts 256 \
    --max-iter 2000 \
    --seed 0 \
    --out-dir results \
    --task-id "${SLURM_ARRAY_TASK_ID}"
