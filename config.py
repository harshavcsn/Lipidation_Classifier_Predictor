"""
config.py -- every knob for the LIPML pipeline lives here.

Edit this file, then run the numbered scripts in order:

    python 01_build_features.py
    python 02_make_jobs.py
    ./run_search.sh 32            # or sbatch run_search.slurm
    python 03_aggregate.py
    python 04_evaluate_best.py
    python 05_ablation.py
    python 06_final_train_predict.py

No held-out test split is used anywhere. Model selection is done with
RepeatedStratifiedKFold over 100% of the labeled data (the k-fold split IS
the train/validation split), and the winning configuration for each model is
then refit on 100% of the data for the final predictions.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Each path can be overridden with an environment variable, so a new dataset
# can be trained into its own output directory without editing this file:
#   LIPML_DATA=my_data.csv LIPML_OUT=results_my_run ./run_all.sh
CSV_PATH = os.environ.get("LIPML_DATA", "list3.csv")                # labeled data: no header, "sequence,label"
NEW_SEQ_CSV_PATH = os.environ.get("LIPML_NEW", "new_predict.csv")   # unlabeled: no header, single sequence column

OUT_DIR = os.environ.get("LIPML_OUT", "results_4block_pca")         # everything is written under here
FEATURE_DIR = os.path.join(OUT_DIR, "features")
JOBS_DIR = os.path.join(OUT_DIR, "jobs")
SHARD_DIR = os.path.join(OUT_DIR, "shards")
FINAL_DIR = os.path.join(OUT_DIR, "final")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ---------------------------------------------------------------------------
# Data / CV
# ---------------------------------------------------------------------------
SEED = 10
N_SPLITS = 4          # auto-reduced if the smallest class has fewer members
N_REPEATS = 20        # repeats of the stratified k-fold, averages out noise
MERGE_CLASSES = False # True -> merge "MS" into "fiber" (2-class problem)

# Metric used to rank hyperparameter configurations.
# One of: "macro_f1", "balanced_accuracy", "accuracy"
SELECTION_METRIC = "macro_f1"

# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------
USE_ONEHOT = False    # 3 positions x 20 amino acids = 60 indicator columns
USE_PCA = True        # the 4-block AAindex PCA lookup described below

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
POSITIONS = ["A2", "A3", "A4"]   # position 1 is the fixed "G" anchor, discarded

# AAindex descriptors are split into 4 categories:
#   nonsec    -- every clean descriptor NOT tagged "sec_struct" by AAindex
#   helix     -- sec_struct descriptors whose description mentions helix/alpha
#   beta      -- sec_struct descriptors mentioning beta/sheet/strand/extended
#   othersec  -- the remaining sec_struct descriptors (turn/coil/bend/loop/...)
# Each category gets its own independent PCA; the resulting per-amino-acid
# score tables are concatenated into one lookup table.
N_PCS = {
    "nonsec": 15,
    "helix": 15,
    "beta": 15,
    "othersec": 15,
}

# How each block's PC scores are rescaled across the 20 amino acids.
# "minmax"  -> each PC squashed to [0, 1]   (default)
# "zscore"  -> each PC standardized to mean 0, std 1
# "raw"     -> untouched PCA scores
PCA_SCALING = "minmax"

# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
# None  -> evaluate the FULL grid for every model.
# int   -> randomly subsample at most this many combinations per model,
#          useful for a quick smoke test before committing the CPU hours.
# Override with LIPML_MAX_COMBOS=<int> (or "none" for the full grid).
_max_combos = os.environ.get("LIPML_MAX_COMBOS", "20000")
MAX_COMBOS_PER_MODEL = None if _max_combos.lower() == "none" else int(_max_combos)

# SVC(probability=True) fits an internal 5-fold Platt calibration on every fit.
# That is slow and unreliable when a class has very few members, so it is off
# during the search. Turn it on only for the final refit if you want
# predict_proba on new sequences.
SVM_PROBABILITY_IN_SEARCH = False
SVM_PROBABILITY_IN_FINAL = False

# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------
N_PERM_REPEATS = 5    # shuffles per feature per fold in permutation importance
RUN_BLOCK_ABLATION = True   # true drop-a-whole-block retraining ablation
TOP_N_FEATURES_PLOT = 20

# The ablation stage is far more expensive per fold than the search: it costs
# one refit plus n_features x N_PERM_REPEATS predictions on every fold. Using
# all N_SPLITS x N_REPEATS folds here buys almost nothing -- the importance
# estimates have long since stabilized -- so the fold list is truncated.
# Set to None to use every fold.
ABLATION_MAX_FOLDS = None

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
# The search is parallelized across PROCESSES (one shard per CPU), so every
# worker must stay single-threaded or the processes will fight over cores and
# the whole thing gets slower. Do not raise this.
THREADS_PER_WORKER = 1
