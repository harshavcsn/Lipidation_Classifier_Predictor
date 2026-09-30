#!/usr/bin/env python3
"""
Step 2 -- enumerate every (model, hyperparameter combination) pair into a
single flat job manifest.

Writing the manifest to disk once, instead of having each worker regenerate
the grid, guarantees that job number 4711 means exactly the same thing in
every process. Workers then claim jobs by striding through this file.

Output: JOBS_DIR/jobs.jsonl  (one JSON object per line)
"""

import json
import os
import random
import warnings

import pandas as pd

import config
import cv
import models


def main():
    os.makedirs(config.JOBS_DIR, exist_ok=True)
    rng = random.Random(config.SEED)

    jobs = []
    print("Expanding hyperparameter grids\n")
    for model_name in models.MODEL_SPECS:
        combos = models.expand_grid(model_name)
        total = len(combos)

        cap = config.MAX_COMBOS_PER_MODEL
        if cap is not None and total > cap:
            combos = rng.sample(combos, cap)
            note = f"  (randomly subsampled to {cap})"
        else:
            note = ""

        print(f"  {model_name:20s} {total:7,d} combinations{note}")
        for params in combos:
            jobs.append({"model": model_name, "params": params})

    by_model = {}
    for job in jobs:
        by_model.setdefault(job["model"], []).append(job)

    # Shuffle before assigning job_ids, so that workers taking every Nth job
    # each get a representative mix of cheap and expensive configurations.
    #
    # Do NOT replace this with a round-robin interleave of the model families.
    # That looks like better balancing but is much worse: it makes the manifest
    # periodic with period 4, so any shard count that is a multiple of 4 sends
    # every random forest to the same unlucky CPU while other workers finish in
    # seconds. A seeded shuffle has no period to alias against, and rerunning
    # this script with the same SEED reproduces the identical manifest.
    rng.shuffle(jobs)

    path = os.path.join(config.JOBS_DIR, "jobs.jsonl")
    with open(path, "w") as fh:
        for job_id, job in enumerate(jobs):
            fh.write(json.dumps({"job_id": job_id, **job}) + "\n")

    total = len(jobs)
    print(f"\n  {total:,d} jobs written to {path}")

    try:
        with open(os.path.join(config.FEATURE_DIR, "meta.json")) as fh:
            n_folds = json.load(fh)["n_folds"]
    except FileNotFoundError:
        n_folds = config.N_SPLITS * config.N_REPEATS
    print(f"  {n_folds} folds per job -> {total * n_folds:,d} model fits in total")

    estimate_runtime(by_model, n_folds, rng)


def estimate_runtime(by_model, n_folds, rng, n_probe=6):
    """Time a handful of real fits per family and extrapolate.

    A flat per-fit rate is useless here: on this data a naive Bayes fit takes
    about 2 ms while a 500-tree forest takes closer to 750 ms. Probing the
    actual grid is the only way to get a number worth planning around.
    """
    import time

    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    try:
        X = pd.read_csv(os.path.join(config.FEATURE_DIR, "X.csv"))
        y = pd.read_csv(os.path.join(config.FEATURE_DIR, "y.csv"))["label"].to_numpy()
    except FileNotFoundError:
        print("\n  (run 01_build_features.py to get a runtime estimate)")
        return

    print("\n  Timing a few real fits per family to estimate the cost")
    total_seconds = 0.0
    for model_name, jobs in by_model.items():
        probes = rng.sample(jobs, min(n_probe, len(jobs)))
        elapsed = 0.0
        for job in probes:
            start = time.perf_counter()
            try:
                cv.fit_one(model_name, job["params"], X.values, y)
            except Exception:
                pass
            elapsed += time.perf_counter() - start

        per_fit = elapsed / len(probes)
        family_seconds = per_fit * n_folds * len(jobs)
        total_seconds += family_seconds
        print(f"    {model_name:20s} ~{per_fit * 1000:7.1f} ms/fit  ->  "
              f"{family_seconds / 3600:8.1f} core-hours")

    core_hours = total_seconds / 3600
    print(f"\n    {'TOTAL':20s} {' ' * 18}{core_hours:8.1f} core-hours")
    for n_cpu in (8, 16, 32, 64):
        print(f"      ~{core_hours / n_cpu:6.1f} h wall-clock on {n_cpu} CPUs")
    print("\n  Probes are a small sample, so treat this as a planning number. "
          "If it is too slow, set MAX_COMBOS_PER_MODEL in config.py and rerun "
          "this script -- the grid is then randomly subsampled.")


if __name__ == "__main__":
    main()
