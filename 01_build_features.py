#!/usr/bin/env python3
"""
Step 1 -- build the AAindex PCA lookup table and the feature matrix, once.

This is the only script that imports the `aaindex` package. Everything it
produces is cached to disk, so the search workers can run on machines where
that package is not installed and, more importantly, so every worker uses
byte-identical features.

Outputs (under FEATURE_DIR):
    pca_table.csv        20 amino acids x N components lookup table
    descriptor_groups.json   which AAindex IDs went into which of the 4 blocks
    X.parquet / X.csv    feature matrix for the labeled sequences
    y.csv                labels, aligned row-for-row with X
    meta.json            fold count, class counts, config snapshot
"""

import json
import os

import pandas as pd

import config
import cv
import features


def main():
    os.makedirs(config.FEATURE_DIR, exist_ok=True)

    print("Loading labeled sequences")
    df = features.load_labeled_data()

    print("\nBuilding the 4-category AAindex PCA lookup table")
    pca_table, groups = features.build_pca_table()

    pca_path = os.path.join(config.FEATURE_DIR, "pca_table.csv")
    pca_table.to_csv(pca_path)
    with open(os.path.join(config.FEATURE_DIR, "descriptor_groups.json"), "w") as fh:
        json.dump({k: list(v) for k, v in groups.items()}, fh, indent=2)

    print("\nEncoding sequences")
    X = features.build_feature_matrix(df, pca_table)
    y = df["label"].to_numpy()

    n_onehot = 60 if config.USE_ONEHOT else 0
    print(f"  Feature matrix: {X.shape[0]} sequences x {X.shape[1]} features "
          f"({n_onehot} one-hot + {X.shape[1] - n_onehot} PCA)")

    X.to_csv(os.path.join(config.FEATURE_DIR, "X.csv"), index=False)
    pd.Series(y, name="label").to_csv(
        os.path.join(config.FEATURE_DIR, "y.csv"), index=False)
    try:
        X.to_parquet(os.path.join(config.FEATURE_DIR, "X.parquet"), index=False)
    except Exception as exc:  # pyarrow is optional; CSV is the fallback
        print(f"  (parquet unavailable, using CSV only: {exc})")

    print("\nCross-validation plan")
    n_splits = cv.resolve_n_splits(y)
    splits = cv.make_splits(X, y, n_splits=n_splits)

    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "classes": sorted(pd.Series(y).unique().tolist()),
        "class_counts": pd.Series(y).value_counts().to_dict(),
        "n_splits": int(n_splits),
        "n_repeats": int(config.N_REPEATS),
        "n_folds": len(splits),
        "seed": int(config.SEED),
        "use_onehot": bool(config.USE_ONEHOT),
        "use_pca": bool(config.USE_PCA),
        "pca_scaling": config.PCA_SCALING,
        "n_pcs": config.N_PCS,
        "merge_classes": bool(config.MERGE_CLASSES),
        "selection_metric": config.SELECTION_METRIC,
        "descriptor_counts": {k: len(v) for k, v in groups.items()},
    }
    with open(os.path.join(config.FEATURE_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nWrote features to {config.FEATURE_DIR}/")


if __name__ == "__main__":
    main()
