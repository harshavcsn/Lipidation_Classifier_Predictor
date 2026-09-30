"""
cv.py -- the cross-validation harness shared by every stage of the pipeline.

There is no held-out test set. The repeated stratified k-fold split IS the
train/validation split: on each of the N_SPLITS x N_REPEATS folds the model is
fit on the training portion and scored on the validation portion it never saw.
Because the fold list is built once from a fixed seed and reused by every
script and every shard, all models and all hyperparameter configurations are
compared on byte-identical folds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import config
import models

METRIC_KEYS = {
    "macro_f1": "macro_f1_mean",
    "balanced_accuracy": "balanced_acc_mean",
    "accuracy": "accuracy_mean",
}


def resolve_n_splits(y, n_splits=None, verbose=True):
    """Shrink n_splits if the rarest class cannot fill that many folds."""
    n_splits = config.N_SPLITS if n_splits is None else n_splits
    min_class = int(pd.Series(y).value_counts().min())
    if n_splits > min_class:
        if verbose:
            print(f"  !! N_SPLITS={n_splits} exceeds the smallest class count "
                  f"({min_class}); reducing to {max(2, min_class)}.")
        n_splits = max(2, min_class)
    return n_splits


def make_splits(X, y, n_splits=None, n_repeats=None, seed=None, verbose=True):
    n_splits = resolve_n_splits(y, n_splits, verbose=verbose)
    n_repeats = config.N_REPEATS if n_repeats is None else n_repeats
    seed = config.SEED if seed is None else seed

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    splits = list(rskf.split(X, y))
    if verbose:
        print(f"  {len(splits)} folds total ({n_splits}-fold x {n_repeats} repeats)")
    return splits


def fit_one(model_name, params, X_tr, y_tr, **build_kwargs):
    """Build a fresh estimator and fit it, applying sample weights if needed."""
    clf = models.build_estimator(model_name, params, **build_kwargs)
    if models.needs_sample_weight(model_name, params):
        sw = compute_sample_weight("balanced", y_tr)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        clf.fit(X_tr, y_tr)
    return clf


def evaluate(model_name, params, X, y, splits, collect_predictions=False):
    """Score one configuration across every fold.

    Returns a dict of mean/std metrics. When collect_predictions is True it
    also returns the pooled validation-fold true/predicted labels, which is
    what the aggregated confusion matrices are built from. Pooled predictions
    are NOT collected during the main search -- storing them for tens of
    thousands of configurations would balloon the shard files for no reason,
    so they are recomputed later for the winning configurations only.
    """
    accs, bal_accs, f1s = [], [], []
    pooled_true, pooled_pred = [], []

    X_values = X.values if hasattr(X, "values") else np.asarray(X)
    y = np.asarray(y)

    for tr_idx, val_idx in splits:
        X_tr, X_val = X_values[tr_idx], X_values[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        clf = fit_one(model_name, params, X_tr, y_tr)
        y_pred = clf.predict(X_val)

        accs.append(accuracy_score(y_val, y_pred))
        bal_accs.append(balanced_accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        if collect_predictions:
            pooled_true.extend(y_val.tolist())
            pooled_pred.extend(y_pred.tolist())

    out = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_acc_mean": float(np.mean(bal_accs)),
        "balanced_acc_std": float(np.std(bal_accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_folds": len(splits),
    }
    if collect_predictions:
        out["pooled_true"] = pooled_true
        out["pooled_pred"] = pooled_pred
    return out


def selection_column():
    """Name of the results column used to rank configurations."""
    try:
        return METRIC_KEYS[config.SELECTION_METRIC]
    except KeyError:
        raise ValueError(
            f"SELECTION_METRIC must be one of {sorted(METRIC_KEYS)}, "
            f"got '{config.SELECTION_METRIC}'"
        )


def rank_results(results_df):
    """Sort best-first: highest score, then lowest fold-to-fold variance."""
    col = selection_column()
    std_col = col.replace("_mean", "_std")
    return (results_df
            .sort_values([col, std_col], ascending=[False, True])
            .reset_index(drop=True))


def best_per_model(results_df):
    """Top-scoring configuration for each model family, best model first.

    drop_duplicates keeps whole rows. A groupby().first() would take the first
    non-null value column by column and could therefore splice together
    parameters from different configurations, since legitimate values like
    max_depth=None arrive as NaN.
    """
    ranked = rank_results(results_df)
    return (ranked
            .drop_duplicates(subset="model", keep="first")
            .reset_index(drop=True))
