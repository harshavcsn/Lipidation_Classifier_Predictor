#!/usr/bin/env python3
"""
Step 5 -- the ablation study, in two complementary forms.

1. Permutation importance (per feature).
   For every CV fold, fit the winning configuration on the fold's training
   rows, then shuffle one feature at a time in the fold's held-out rows and
   measure how far macro-F1 falls. Importances are averaged over all folds and
   repeats. Shuffling is the standard cheap stand-in for a true per-feature
   ablation: it answers "how much does this feature matter?" without retraining
   once per dropped feature, and it works the same way for tree-based and
   distance-based models alike.

2. Block ablation (per feature group), enabled by RUN_BLOCK_ABLATION.
   This one is a genuine ablation -- whole groups of columns are removed and
   the model is refit from scratch on the same folds. Groups are the three
   sequence positions (A2, A3, A4), the four AAindex PCA categories (nonsec,
   helix, beta, othersec) plus one-hot if enabled, and every position x
   category intersection. This is the version that addresses the central
   question: does the helix block carry information the beta block does not?

With a dataset this small, treat both rankings as suggestive rather than
settled. Rerun with a different SEED and see which conclusions survive.

Outputs:
    FINAL_DIR/permutation_importance.csv
    FINAL_DIR/consensus_importance.csv
    FINAL_DIR/block_ablation.csv
    FIG_DIR/permutation_importance_per_model.png
    FIG_DIR/permutation_importance_consensus.png
    FIG_DIR/block_ablation.png
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

import config  # noqa: E402
import cv  # noqa: E402
import features  # noqa: E402
from run_shard import load_features  # noqa: E402

BLOCK_COLORS = {
    "onehot": "#4C72B0",
    "nonsec": "#DD8452",
    "helix": "#55A868",
    "beta": "#C44E52",
    "othersec": "#8172B2",
}


def feature_colors(columns):
    return [BLOCK_COLORS.get(features.feature_group(c)[1], "#937860")
            for c in columns]


# ---------------------------------------------------------------------------
# 1. Permutation importance
# ---------------------------------------------------------------------------
def _one_fold_importance(model_name, params, X_values, y, tr_idx, val_idx):
    clf = cv.fit_one(model_name, params, X_values[tr_idx], y[tr_idx])
    perm = permutation_importance(
        clf, X_values[val_idx], y[val_idx],
        scoring="f1_macro",
        n_repeats=config.N_PERM_REPEATS,
        random_state=config.SEED,
        n_jobs=1,          # parallelism is across folds, not within a fold
    )
    return perm.importances_mean


def permutation_importances(best, X, y, splits, n_jobs):
    X_values = X.values
    y = np.asarray(y)
    out = {}

    for _, row in best.iterrows():
        model_name, params = row["model"], row["params"]
        print(f"\n  {model_name}: {X.shape[1]} features x {len(splits)} folds "
              f"x {config.N_PERM_REPEATS} shuffles", flush=True)
        start = time.time()

        fold_scores = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_one_fold_importance)(
                model_name, params, X_values, y, tr_idx, val_idx)
            for tr_idx, val_idx in splits
        )

        out[model_name] = pd.Series(
            np.mean(fold_scores, axis=0), index=X.columns
        ).sort_values(ascending=False)
        print(f"    done in {(time.time() - start) / 60:.1f} min", flush=True)

    return out


def plot_permutation(importances, path):
    n = len(importances)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 4.6 * n))
    if n == 1:
        axes = [axes]

    top_n = config.TOP_N_FEATURES_PLOT
    for ax, (model_name, imp) in zip(axes, importances.items()):
        top = imp.head(top_n)
        colors = feature_colors(top.index)
        ax.barh(list(top.index)[::-1], list(top.values)[::-1],
                color=colors[::-1])
        ax.set_xlabel("Drop in macro-F1 when the feature is shuffled")
        ax.set_title(f"Top {top_n} features -- {model_name}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in BLOCK_COLORS.values()]
    fig.legend(handles, list(BLOCK_COLORS.keys()),
               loc="lower center", ncol=len(BLOCK_COLORS), frameon=False)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(path, dpi=150)
    plt.close()


def consensus(importances, path_csv, path_fig):
    """Average the per-model rankings after putting them on a common scale.

    Each model's raw importances live on its own scale -- an F1 drop of 0.02
    can be large for one algorithm and trivial for another -- so each series is
    min-max normalized to [0, 1] before averaging. Without that step whichever
    model happens to produce the biggest raw numbers would dominate.
    """
    normed = {}
    for model_name, imp in importances.items():
        span = imp.max() - imp.min()
        normed[model_name] = (imp - imp.min()) / span if span > 0 else imp * 0.0

    table = pd.DataFrame(normed)
    table["consensus"] = table.mean(axis=1)
    table = table.sort_values("consensus", ascending=False)
    table.to_csv(path_csv)

    top = table["consensus"].head(config.TOP_N_FEATURES_PLOT)
    colors = feature_colors(top.index)
    plt.figure(figsize=(7.5, 6.5))
    plt.barh(list(top.index)[::-1], list(top.values)[::-1], color=colors[::-1])
    plt.xlabel("Mean normalized importance across models (0-1 per model)")
    plt.title("Consensus feature importance across all model families")
    plt.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in BLOCK_COLORS.values()]
    plt.legend(handles, list(BLOCK_COLORS.keys()), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(path_fig, dpi=150)
    plt.close()
    return table


# ---------------------------------------------------------------------------
# 2. Block ablation
# ---------------------------------------------------------------------------
def define_blocks(columns):
    """Column groups to drop: positions, categories, and their intersections."""
    groups = {}
    parsed = {c: features.feature_group(c) for c in columns}

    for pos in sorted({p for p, _ in parsed.values()}):
        groups[f"position:{pos}"] = [c for c, (p, _) in parsed.items() if p == pos]

    for block in sorted({b for _, b in parsed.values()}):
        groups[f"category:{block}"] = [c for c, (_, b) in parsed.items() if b == block]

    for pos in sorted({p for p, _ in parsed.values()}):
        for block in sorted({b for _, b in parsed.values()}):
            cols = [c for c, (p, b) in parsed.items() if p == pos and b == block]
            if cols:
                groups[f"{pos} x {block}"] = cols

    return groups


def _ablate_one(model_name, params, X, y, splits, block_name, cols):
    kept = [c for c in X.columns if c not in set(cols)]
    if not kept:
        return None
    res = cv.evaluate(model_name, params, X[kept], y, splits)
    return {"model": model_name, "block": block_name,
            "n_features_dropped": len(cols),
            "macro_f1_mean": res["macro_f1_mean"],
            "macro_f1_std": res["macro_f1_std"]}


def block_ablation(best, X, y, splits, n_jobs):
    blocks = define_blocks(X.columns)
    rows = []

    for _, row in best.iterrows():
        model_name, params = row["model"], row["params"]
        print(f"\n  {model_name}: refitting with each of {len(blocks)} blocks "
              f"removed, over {len(splits)} folds", flush=True)
        start = time.time()

        baseline = cv.evaluate(model_name, params, X, y, splits)
        base_f1 = baseline["macro_f1_mean"]
        rows.append({"model": model_name, "block": "(none -- baseline)",
                     "n_features_dropped": 0,
                     "macro_f1_mean": base_f1,
                     "macro_f1_std": baseline["macro_f1_std"],
                     "delta_vs_baseline": 0.0})

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_ablate_one)(model_name, params, X, y, splits, name, cols)
            for name, cols in blocks.items()
        )

        for res in results:
            if res is None:
                continue
            res["delta_vs_baseline"] = res["macro_f1_mean"] - base_f1
            rows.append(res)

        print(f"    baseline macro-F1 {base_f1:.4f}, "
              f"done in {(time.time() - start) / 60:.1f} min")
        for res in sorted((r for r in results if r),
                          key=lambda r: r["delta_vs_baseline"]):
            print(f"      drop {res['block']:24s} ({res['n_features_dropped']:3d} "
                  f"cols) -> macro-F1 {res['macro_f1_mean']:.4f}  "
                  f"({res['delta_vs_baseline']:+.4f})")

    return pd.DataFrame(rows)


def plot_block_ablation(table, path):
    real = table[table["block"] != "(none -- baseline)"]
    order = (real.groupby("block")["delta_vs_baseline"].mean()
             .sort_values().index.tolist())
    model_names = table["model"].unique().tolist()

    fig, axes = plt.subplots(len(model_names), 1,
                             figsize=(8.5, 0.32 * len(order) * len(model_names) + 2),
                             squeeze=False)

    for ax, model_name in zip(axes[:, 0], model_names):
        sub = real[real["model"] == model_name].set_index("block").reindex(order)
        colors = ["#C44E52" if v < 0 else "#55A868"
                  for v in sub["delta_vs_baseline"]]
        ax.barh(sub.index, sub["delta_vs_baseline"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Change in macro-F1 when the block is removed "
                      "(negative = the block was helping)")
        ax.set_title(f"Block ablation -- {model_name}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="cores for permutation importance (this script runs alone)")
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()

    os.makedirs(config.FINAL_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)

    best = pd.read_pickle(os.path.join(config.OUT_DIR, "best_per_model.pkl"))
    X, y = load_features()
    splits = cv.make_splits(X, y)

    cap = config.ABLATION_MAX_FOLDS
    if cap is not None and len(splits) > cap:
        print(f"  Using the first {cap} of {len(splits)} folds for ablation "
              f"(ABLATION_MAX_FOLDS in config.py)")
        splits = splits[:cap]

    if not args.skip_permutation:
        print("Permutation importance")
        importances = permutation_importances(best, X, y, splits, args.n_jobs)

        pd.DataFrame(importances).to_csv(
            os.path.join(config.FINAL_DIR, "permutation_importance.csv"))
        plot_permutation(
            importances,
            os.path.join(config.FIG_DIR, "permutation_importance_per_model.png"))
        table = consensus(
            importances,
            os.path.join(config.FINAL_DIR, "consensus_importance.csv"),
            os.path.join(config.FIG_DIR, "permutation_importance_consensus.png"))
        print("\n  Top 10 consensus features:")
        print(table["consensus"].head(10).to_string())

    if config.RUN_BLOCK_ABLATION:
        print("\n\nBlock ablation (refitting with whole groups removed)")
        table = block_ablation(best, X, y, splits, args.n_jobs)
        table.to_csv(os.path.join(config.FINAL_DIR, "block_ablation.csv"),
                     index=False)
        plot_block_ablation(
            table, os.path.join(config.FIG_DIR, "block_ablation.png"))

    print(f"\nWrote ablation results to {config.FINAL_DIR}/")
    print("Next: python3 06_final_train_predict.py")


if __name__ == "__main__":
    main()
