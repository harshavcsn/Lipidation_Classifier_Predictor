#!/usr/bin/env bash
#
# Launch the hyperparameter search across N single-CPU worker processes.
#
#   ./run_search.sh            # uses every core the machine reports
#   ./run_search.sh 32         # 32 workers
#   ./run_search.sh 32 --resume
#
# Each worker gets its own output directory under results/shards/, so nothing
# is shared and nothing needs locking. Workers are pinned to one thread each;
# oversubscribing BLAS threads on top of process parallelism makes the whole
# search slower.
#
# On a cluster, pass the core count explicitly. `nproc` may report the whole
# machine rather than your allocation, and launching 128 workers inside a
# 32-core reservation is slower than launching 32.

set -euo pipefail

N_SHARDS="${1:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
shift || true

# Collect any remaining flags (e.g. --resume) to hand to every worker.
# Bash before 4.4 treats "${ARR[@]}" as unbound under `set -u` when the array
# is empty, so the expansion below is guarded rather than written directly.
EXTRA_ARGS=()
if [ "$#" -gt 0 ]; then
    EXTRA_ARGS=("$@")
fi
extra_args() {
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        printf '%s\n' "${EXTRA_ARGS[@]}"
    fi
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"

echo "Launching ${N_SHARDS} single-CPU workers"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "Worker flags: $(extra_args | tr '\n' ' ')"
fi
echo

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    seq 0 $((N_SHARDS - 1)) | xargs -P "${N_SHARDS}" -I {} \
        "${PYTHON}" run_shard.py --shard-id {} --n-shards "${N_SHARDS}" "${EXTRA_ARGS[@]}"
else
    seq 0 $((N_SHARDS - 1)) | xargs -P "${N_SHARDS}" -I {} \
        "${PYTHON}" run_shard.py --shard-id {} --n-shards "${N_SHARDS}"
fi

echo
echo "All shards finished. Next: python3 03_aggregate.py"