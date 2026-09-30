"""
models.py -- the four model families and their hyperparameter grids.

Written against the scikit-learn 1.8 / 1.9 API. Two changes from older code
are worth calling out, because they affect this pipeline directly:

  * LogisticRegression's `penalty` argument is deprecated (removed in 1.10).
    The penalty family is now expressed through `l1_ratio`:
        l1_ratio = 0            ->  pure L2      (Ridge logistic regression)
        l1_ratio = 1            ->  pure L1      (Lasso logistic regression)
        0 < l1_ratio < 1        ->  Elastic Net
    Only the "saga" solver supports the full range, so all three families are
    swept in a single grid over (C, l1_ratio) with solver="saga".

  * LogisticRegression's `n_jobs` is deprecated and has no effect, and
    `multi_class` is gone -- multiclass problems automatically use the full
    multinomial loss for every solver except liblinear.

Every `build` function takes a plain dict of JSON-serializable parameters and
returns a fresh, unfitted estimator.
"""

from itertools import product

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def _floats(arr):
    """numpy scalars are not JSON-serializable; plain floats are."""
    return [float(v) for v in arr]


# ---------------------------------------------------------------------------
# 1. Regularized logistic regression: Ridge + Lasso + Elastic Net in one sweep
# ---------------------------------------------------------------------------
def _build_logistic(p):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=p["C"],
            l1_ratio=p["l1_ratio"],   # 0 = ridge, 1 = lasso, between = elasticnet
            solver="saga",            # the only solver covering all l1_ratio
            class_weight=p["class_weight"],
            fit_intercept=True,
            max_iter=20000,
            tol=1e-4,
            random_state=config.SEED,
        )),
    ])


def _describe_logistic(p):
    r = p["l1_ratio"]
    if r == 0.0:
        family = "Ridge (L2)"
    elif r == 1.0:
        family = "Lasso (L1)"
    else:
        family = "ElasticNet"
    return {"penalty_family": family}


# ---------------------------------------------------------------------------
# 2. Gaussian Naive Bayes
# ---------------------------------------------------------------------------
def _build_nb(p):
    # GaussianNB has no class_weight argument, so class balancing is done by
    # passing per-sample weights at fit time. The "balance" flag below turns
    # that on or off so the search can decide which works better here.
    return GaussianNB(var_smoothing=p["var_smoothing"])


# ---------------------------------------------------------------------------
# 3. Support vector machine, RBF kernel
# ---------------------------------------------------------------------------
def _build_svm(p, probability=None):
    if probability is None:
        probability = config.SVM_PROBABILITY_IN_SEARCH

    # `probability` is deprecated in sklearn 1.9 and removed in 1.11, and it
    # warns whenever it is passed at all -- even as False. So it is never
    # passed. When probabilities are actually wanted the SVC is wrapped in
    # CalibratedClassifierCV, which is the documented replacement.
    svc = SVC(
        kernel="rbf",
        C=p["C"],
        gamma=p["gamma"],
        class_weight=p["class_weight"],
        cache_size=200,
        random_state=config.SEED,
    )

    if probability:
        # cv=3 rather than the default 5: the calibration split is stratified,
        # so with only a handful of sequences in the rarest class a 5-fold
        # split can leave a fold with no members of that class.
        clf = CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=3)
    else:
        clf = svc

    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# 4. Random forest
# ---------------------------------------------------------------------------
def _build_rf(p):
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        criterion=p["criterion"],
        max_features=p["max_features"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        min_samples_split=p["min_samples_split"],
        max_leaf_nodes=p["max_leaf_nodes"],
        ccp_alpha=p["ccp_alpha"],
        max_samples=p["max_samples"],
        class_weight=p["class_weight"],
        bootstrap=True,          # required for max_samples / balanced_subsample
        n_jobs=1,                # parallelism happens across shards, not here
        random_state=config.SEED,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "ElasticNetLogistic": {
        "build": _build_logistic,
        "describe": _describe_logistic,
        "needs_sample_weight": lambda p: False,
        "grid": {
            "C": _floats(np.logspace(-4, 4, 17)),
            "l1_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9, 1.0],
            "class_weight": ["balanced", None],
        },
    },

    # "ElasticNetLogistic": {
    #     "build": _build_logistic,
    #     "describe": _describe_logistic,
    #     "needs_sample_weight": lambda p: False,
    #     "grid": {
    #         "C": _floats(np.logspace(-4, 4, 2)),
    #         "l1_ratio": [0.0,  1.0],
    #         "class_weight": ["balanced", None],
    #     },
    # },

    # Gaussian naive Bayes is disabled: it scored well below the other three
    # families (macro-F1 ~0.43) and is not part of the released models.
    # Uncomment this entry to include it in the search and final refit.
    # "NaiveBayes": {
    #     "build": _build_nb,
    #     "describe": lambda p: {},
    #     "needs_sample_weight": lambda p: bool(p["balance"]),
    #     "grid": {
    #         "var_smoothing": _floats(np.logspace(-12, 0, 25)),
    #         "balance": [True, False],
    #     },
    # },

    "SVM_RBF": {
        "build": _build_svm,
        "describe": lambda p: {},
        "needs_sample_weight": lambda p: False,
        "grid": {
            "C": _floats(np.logspace(-3, 5, 17)),
            "gamma": ["scale", "auto"] + _floats(np.logspace(-6, 2, 17)),
            "class_weight": ["balanced", None],
        },
    },

    # The random forest grid dominates the total runtime -- a 500-tree fit is
    # roughly 400x slower than a naive Bayes fit on data this size -- so the
    # axes below were chosen to avoid spending hours on redundant distinctions.
    # min_samples_split is fixed at 2 because min_samples_leaf already
    # constrains leaf size and, with only ~64 sequences, the two knobs select
    # nearly identical trees. max_leaf_nodes stops at 8 for the same reason:
    # max_depth already caps tree complexity from the other direction.
    # Widen any of these if you have the CPU hours to spare.
    "RandomForest": {
        "build": _build_rf,
        "describe": lambda p: {},
        "needs_sample_weight": lambda p: False,
        "grid": {
            "n_estimators": [50,100,200,500],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", 0.3, 0.5, 1.0],
            "max_depth": [2, 4, 8, 10, 50, 100, None],
            "min_samples_leaf": [1, 2, 4,10,20],
            "min_samples_split": [2,3],
            "max_leaf_nodes": [None, 8,10,20],
            "ccp_alpha": [0.0, 0.01, 0.1],
            "max_samples": [0.5,0.6,0.8, None],
            "class_weight": ["balanced_subsample", "balanced"],
        },
    },

    # "RandomForest": {
    #     "build": _build_rf,
    #     "describe": lambda p: {},
    #     "needs_sample_weight": lambda p: False,
    #     "grid": {
    #         "n_estimators": [50,100],
    #         "criterion": ["gini",],
    #         "max_features": [ 0.5],
    #         "max_depth": [ None],
    #         "min_samples_leaf": [1],
    #         "min_samples_split": [2],
    #         "max_leaf_nodes": [None, ],
    #         "ccp_alpha": [0.0, 0.01, 0.1],
    #         "max_samples": [None],
    #         "class_weight": ["balanced_subsample"],
    #     },
    # },    
}


def expand_grid(model_name):
    """All parameter combinations for one model, in a deterministic order."""
    grid = MODEL_SPECS[model_name]["grid"]
    keys = sorted(grid.keys())
    return [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]


def build_estimator(model_name, params, **kwargs):
    return MODEL_SPECS[model_name]["build"](params, **kwargs)


def needs_sample_weight(model_name, params):
    return MODEL_SPECS[model_name]["needs_sample_weight"](params)


def describe(model_name, params):
    return MODEL_SPECS[model_name]["describe"](params)