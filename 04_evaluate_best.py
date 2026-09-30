#!/usr/bin/env python3
"""
Step 4 -- pooled cross-validation predictions for the winning configurations.

The search deliberately did not store per-fold predictions; keeping them for
tens of thousands of configurations would bloat every shard file. Here the
winners are re-run over the same folds so their predictions can be pooled into
confusion matrices and per-class reports.

Note what the pooled counts mean: each sequence appears once per repeat, so
the totals are n_sequences x N_REPEATS, not n_sequences. The matrices describe
how often each sequence is classified correctly when it is held out, averaged
over many different training subsets -- not performance on unseen data.

Outputs:
    FINAL_DIR/cv_predictions_<model>.csv
    FINAL_DIR/classification_reports.txt
    FIG_DIR/confusion_matrices_counts.png
    FIG_DIR/confusion_matrices_recall.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config  # noqa: E402
import cv  # noqa: E402
from run_shard import load_features  # noqa: E402


def plot_matrices(matrices, labels, titles, normalize, path):
    n = len(matrices)
    fig, axes = plt.subplots(n, 1, figsize=(5.2, 4.6 * n))
    if n == 1:
        axes = [axes]

    for ax, cm, title in zip(axes, matrices, titles):
        shown = cm.astype(float)
        if normalize:
            with np.errstate(invalid="ignore", divide="ignore"):
                shown = shown / shown.sum(axis=1, keepdims=True)
            shown = np.nan_to_num(shown)
            im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        else:
            im = ax.imshow(shown, cmap="Blues")

        for i in range(len(labels)):
            for j in range(len(labels)):
                text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j]):,d}"
                threshold = 0.5 if normalize else shown.max() / 2
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if shown[i, j] > threshold else "black")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(config.FINAL_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)

    best = pd.read_pickle(os.path.join(config.OUT_DIR, "best_per_model.pkl"))
    X, y = load_features()
    splits = cv.make_splits(X, y)
    labels = sorted(pd.Series(y).unique().tolist())

    matrices, titles, report_lines = [], [], []

    for _, row in best.iterrows():
        model_name, params = row["model"], row["params"]
        print(f"\nRe-running {model_name} over {len(splits)} folds")

        res = cv.evaluate(model_name, params, X, y, splits,
                          collect_predictions=True)
        true, pred = res["pooled_true"], res["pooled_pred"]

        pd.DataFrame({"true": true, "predicted": pred}).to_csv(
            os.path.join(config.FINAL_DIR, f"cv_predictions_{model_name}.csv"),
            index=False)

        acc = accuracy_score(true, pred)
        bal = balanced_accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="macro", zero_division=0)
        print(f"  pooled accuracy {acc:.3f} | balanced {bal:.3f} | macro-F1 {f1:.3f}")

        matrices.append(confusion_matrix(true, pred, labels=labels))
        titles.append(
            f"{model_name}\nacc {acc:.3f} | balanced {bal:.3f} | macro-F1 {f1:.3f}\n"
            f"{len(true):,d} pooled predictions "
            f"({len(y)} sequences x {config.N_REPEATS} repeats)"
        )

        report_lines.append(f"{'=' * 70}\n{model_name}\n{'=' * 70}")
        report_lines.append(f"parameters: {params}\n")
        report_lines.append(classification_report(true, pred, labels=labels,
                                                  zero_division=0))
        report_lines.append("")

    report_path = os.path.join(config.FINAL_DIR, "classification_reports.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines))

    plot_matrices(matrices, labels, titles, normalize=False,
                  path=os.path.join(config.FIG_DIR, "confusion_matrices_counts.png"))
    plot_matrices(matrices, labels, titles, normalize=True,
                  path=os.path.join(config.FIG_DIR, "confusion_matrices_recall.png"))

    print(f"\nWrote reports to {report_path}")
    print("Next: python3 05_ablation.py")


if __name__ == "__main__":
    main()
