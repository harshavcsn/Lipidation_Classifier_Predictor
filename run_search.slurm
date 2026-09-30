#!/usr/bin/env bash
#SBATCH --job-name=lipml-search
#SBATCH --array=0-31              # 32 shards -> must match N_SHARDS below
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/shard_%a.out
#SBATCH --error=slurm_logs/shard_%a.err
#
# SLURM alternative to run_search.sh. Each array task is one shard on one CPU.
#
#   mkdir -p slurm_logs
#   sbatch run_search.slurm
#
# If you change --array above, change N_SHARDS here to match.

set -euo pipefail

N_SHARDS=32

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

python3 run_shard.py \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --resume
