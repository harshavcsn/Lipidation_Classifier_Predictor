#!/usr/bin/env bash
#
# Run the full training pipeline end to end.
#
#   ./run_all.sh          # 32 workers
#   ./run_all.sh 8        # 8 workers
#
# Paths and grid size can be set with LIPML_DATA, LIPML_OUT and
# LIPML_MAX_COMBOS (see config.py).
set -euo pipefail
N_WORKERS="${1:-32}"
mkdir -p logs
python3 01_build_features.py                  2>&1 | tee logs/01_build_features.log
python3 02_make_jobs.py                       2>&1 | tee logs/02_make_jobs.log
./run_search.sh "${N_WORKERS}"                2>&1 | tee logs/run_search.log
python3 03_aggregate.py                       2>&1 | tee logs/03_aggregate.log
python3 04_evaluate_best.py                   2>&1 | tee logs/04_evaluate_best.log
python3 05_ablation.py --n-jobs "${N_WORKERS}" 2>&1 | tee logs/05_ablation.log
python3 06_final_train_predict.py             2>&1 | tee logs/06_final.log
echo "PIPELINE COMPLETE"
