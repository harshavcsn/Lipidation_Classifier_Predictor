#!/usr/bin/env python3
"""
Search worker -- evaluates one shard of the job manifest on a single CPU.

    python run_shard.py --shard-id 0 --n-shards 32

Shard K takes every job whose job_id satisfies  job_id % n_shards == K.
Striding rather than slicing matters: the manifest interleaves model families,
so striding gives each worker a comparable mix of cheap naive-Bayes fits and
expensive random-forest fits instead of one worker inheriting all the slow
ones.

Each shard writes to its own directory, so no two processes ever touch the
same file and there is nothing to lock. Results are flushed to disk
periodically, so a shard that dies partway through still leaves usable output,
and rerunning it with --resume skips the jobs it already finished.
"""

# Thread limits must be set before numpy or scikit-learn are imported, since
# the underlying BLAS reads them at load time. One thread per worker: the
# parallelism here is across processes, and letting each of 32 workers spawn
# 4 BLAS threads on a 32-core box makes everything slower, not faster.
import os

_THREADS = str(os.environ.get("LIPML_THREADS", "1"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import config  # noqa: E402
import cv  # noqa: E402
import models  # noqa: E402


def load_features():
    parquet = os.path.join(config.FEATURE_DIR, "X.parquet")
    csv = os.path.join(config.FEATURE_DIR, "X.csv")
    if os.path.exists(parquet):
        try:
            X = pd.read_parquet(parquet)
        except Exception:
            X = pd.read_csv(csv)
    else:
        X = pd.read_csv(csv)
    y = pd.read_csv(os.path.join(config.FEATURE_DIR, "y.csv"))["label"].to_numpy()
    return X, y


def load_jobs(shard_id, n_shards):
    path = os.path.join(config.JOBS_DIR, "jobs.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No job manifest at {path}. Run 02_make_jobs.py first.")
    jobs = []
    with open(path) as fh:
        for line in fh:
            job = json.loads(line)
            if job["job_id"] % n_shards == shard_id:
                jobs.append(job)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip jobs already present in this shard's output")
    ap.add_argument("--flush-every", type=int, default=25,
                    help="write partial results to disk every N jobs")
    args = ap.parse_args()

    if not 0 <= args.shard_id < args.n_shards:
        raise SystemExit(f"--shard-id must be in [0, {args.n_shards}).")

    out_dir = os.path.join(config.SHARD_DIR, f"shard_{args.shard_id:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.pkl")
    log_path = os.path.join(out_dir, "log.txt")

    # A saga logistic regression on a rank-deficient corner of the grid will
    # not always converge, and an all-one-class fold makes some metrics
    # undefined. Both are expected and would otherwise flood the logs.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*divide by zero.*")

    X, y = load_features()
    splits = cv.make_splits(X, y, verbose=False)
    jobs = load_jobs(args.shard_id, args.n_shards)

    done = set()
    rows = []
    if args.resume and os.path.exists(out_path):
        prev = pd.read_pickle(out_path)
        rows = prev.to_dict("records")
        done = set(prev["job_id"].tolist())

    todo = [j for j in jobs if j["job_id"] not in done]

    def log(msg):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] shard {args.shard_id}: {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    log(f"{len(jobs)} jobs assigned, {len(done)} already done, "
        f"{len(todo)} to run, {len(splits)} folds each")

    t_start = time.time()
    for i, job in enumerate(todo, start=1):
        model_name, params = job["model"], job["params"]
        try:
            res = cv.evaluate(model_name, params, X, y, splits,
                              collect_predictions=False)
            row = {
                "job_id": job["job_id"],
                "model": model_name,
                "params": params,
                "params_json": json.dumps(params, sort_keys=True),
                **models.describe(model_name, params),
                **res,
                "failed": False,
                "error": "",
            }
        except Exception as exc:
            # One bad corner of a grid should never take down a whole shard.
            row = {
                "job_id": job["job_id"],
                "model": model_name,
                "params": params,
                "params_json": json.dumps(params, sort_keys=True),
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            log(f"job {job['job_id']} ({model_name}) failed -- {row['error']}")

        rows.append(row)

        if i % args.flush_every == 0 or i == len(todo):
            pd.DataFrame(rows).to_pickle(out_path)
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed else 0
            remaining = (len(todo) - i) / rate if rate else 0
            log(f"{i}/{len(todo)} done, {rate:.2f} jobs/s, "
                f"~{remaining / 60:.1f} min remaining")

    pd.DataFrame(rows).to_pickle(out_path)
    log(f"finished in {(time.time() - t_start) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
