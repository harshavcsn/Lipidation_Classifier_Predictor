#!/usr/bin/env python3
"""
Step 3 -- merge every shard's results into one table, rank them, and plot the
model comparison.

Outputs:
    OUT_DIR/results_all.pkl / .csv     every configuration that was evaluated
    OUT_DIR/best_per_model.csv         winning configuration per model family
    OUT_DIR/best_per_model.json        the same, as clean parameter dicts
    FIG_DIR/model_comparison_best.png  best score per model family
    FIG_DIR/model_comparison_spread.png  sensitivity to hyperparameter choice
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import cv  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def load_all_shards():
    paths = sorted(glob.glob(os.path.join(config.SHARD_DIR, "shard_*", "results.pkl")))
    if not paths:
        raise FileNotFoundError(
            f"No shard results under {config.SHARD_DIR}/. Run the search first."
        )
    frames = [pd.read_pickle(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Merged {len(paths)} shards -> {len(df):,d} configurations")
    return df


def boxplot_labels(ax_kwargs, labels):
    """matplotlib renamed boxplot's `labels` to `tick_labels` in 3.9."""
    version = tuple(int(v) for v in matplotlib.__version__.split(".")[:2])
    key = "tick_labels" if version >= (3, 9) else "labels"
    ax_kwargs[key] = labels
    return ax_kwargs


def main():
    os.makedirs(config.FIG_DIR, exist_ok=True)

    print("Loading shard results")
    df = load_all_shards()

    n_dup = int(df["job_id"].duplicated().sum())
    if n_dup:
        print(f"  !! {n_dup} duplicate job_ids found; keeping the first of each. "
              f"This usually means shards were run twice with different "
              f"--n-shards values.")
        df = df.drop_duplicates(subset="job_id", keep="first")

    failed = df[df.get("failed", False) == True]  # noqa: E712
    if len(failed):
        print(f"  !! {len(failed)} configurations errored out and are excluded.")
        print(failed.groupby("model")["error"].first().to_string())
        failed.to_csv(os.path.join(config.OUT_DIR, "failed_configs.csv"), index=False)
    df = df[df.get("failed", False) != True].copy()  # noqa: E712

    if df.empty:
        raise SystemExit("Every configuration failed -- nothing to aggregate.")

    metric = cv.selection_column()
    ranked = cv.rank_results(df)
    ranked.to_pickle(os.path.join(config.OUT_DIR, "results_all.pkl"))
    ranked.drop(columns=["params"]).to_csv(
        os.path.join(config.OUT_DIR, "results_all.csv"), index=False)

    display_cols = ["model", "accuracy_mean", "balanced_acc_mean",
                    "macro_f1_mean", "macro_f1_std", "params_json"]
    print(f"\nTop 15 configurations overall (ranked by {metric}):")
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(ranked[display_cols].head(15).to_string(index=False))

    best = cv.best_per_model(df)
    best.to_pickle(os.path.join(config.OUT_DIR, "best_per_model.pkl"))
    best.drop(columns=["params"]).to_csv(
        os.path.join(config.OUT_DIR, "best_per_model.csv"), index=False)

    with open(os.path.join(config.OUT_DIR, "best_per_model.json"), "w") as fh:
        json.dump(
            [
                {
                    "model": row["model"],
                    "params": row["params"],
                    "accuracy_mean": row["accuracy_mean"],
                    "balanced_acc_mean": row["balanced_acc_mean"],
                    "macro_f1_mean": row["macro_f1_mean"],
                    "macro_f1_std": row["macro_f1_std"],
                }
                for _, row in best.iterrows()
            ],
            fh, indent=2, default=str,
        )

    print("\nBest configuration per model family:")
    with pd.option_context("display.width", 200, "display.max_colwidth", 110):
        print(best[display_cols].to_string(index=False))

    # --- bar chart: best score per family -----------------------------------
    std_col = metric.replace("_mean", "_std")
    plt.figure(figsize=(8, 5))
    plt.bar(best["model"], best[metric], yerr=best[std_col], capsize=4,
            color=PALETTE[:len(best)])
    plt.ylabel(f"{metric} (best configuration, mean +/- std across folds)")
    plt.title("Best score per model family (repeated stratified CV, full data)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "model_comparison_best.png"), dpi=150)
    plt.close()

    # --- boxplot: how sensitive is each family to its hyperparameters? -------
    order = best["model"].tolist()
    data = [ranked.loc[ranked["model"] == m, metric].values for m in order]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, **boxplot_labels({}, order))
    plt.ylabel(f"{metric} (one point per configuration)")
    plt.title("Score across the whole hyperparameter grid, by model family")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "model_comparison_spread.png"), dpi=150)
    plt.close()

    print(f"\nWrote tables to {config.OUT_DIR}/ and figures to {config.FIG_DIR}/")
    print("Next: python3 04_evaluate_best.py")


if __name__ == "__main__":
    main()
