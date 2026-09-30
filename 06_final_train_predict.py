#!/usr/bin/env python3
"""
Step 6 -- refit each winning configuration on 100% of the labeled data, save
the fitted models and their learned parameters, and predict the new sequences.

This is the deployment step. The cross-validation above was for choosing
hyperparameters; now that they are chosen, throwing away part of the data
would only make the final model worse, so every labeled sequence is used.

The new sequences are encoded with the cached PCA lookup table from step 1 --
the same table, not a refit one. That matters: the table is derived from
universal AAindex descriptors rather than from your labels, so reusing it is
what keeps the new-sequence features on exactly the same scale as the
training features.

Outputs:
    FINAL_DIR/model_<name>.joblib          fitted estimator
    FINAL_DIR/hyperparameters.json         chosen hyperparameters per model
    FINAL_DIR/learned_parameters.txt       coefficients, class priors, etc.
    FINAL_DIR/logistic_coefficients.csv    per-class coefficients if applicable
    FINAL_DIR/rf_feature_importances.csv   impurity importances if applicable
    FINAL_DIR/new_sequence_predictions.csv predictions + consensus vote
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import config
import cv
import features
import models
from run_shard import load_features


def describe_fitted(model_name, clf, feature_names, class_labels, out_dir):
    """Pull the interesting learned parameters out of a fitted estimator."""
    lines = [f"{'=' * 70}", model_name, "=" * 70]
    estimator = clf.named_steps["clf"] if hasattr(clf, "named_steps") else clf

    # When probabilities are enabled the SVC is wrapped in
    # CalibratedClassifierCV, so the interesting attributes sit one level down.
    if hasattr(estimator, "calibrated_classifiers_"):
        inner = estimator.calibrated_classifiers_[0]
        estimator = getattr(inner, "estimator", inner)
        lines.append("(SVC wrapped in CalibratedClassifierCV for predict_proba; "
                     "the figures below describe the underlying SVC)")

    if hasattr(estimator, "coef_"):
        coefs = pd.DataFrame(
            estimator.coef_,
            columns=feature_names,
            index=(class_labels if len(estimator.coef_) == len(class_labels)
                   else [f"class_{i}" for i in range(len(estimator.coef_))]),
        )
        coefs.insert(0, "intercept", estimator.intercept_)
        coefs.T.to_csv(os.path.join(out_dir, f"coefficients_{model_name}.csv"))

        n_zero = int((estimator.coef_ == 0).sum())
        total = int(estimator.coef_.size)
        lines.append(f"coefficient matrix: {estimator.coef_.shape}")
        lines.append(f"exactly-zero coefficients: {n_zero}/{total} "
                     f"({n_zero / total:.1%}) -- L1 shrinkage drives these to zero")
        lines.append("\nlargest absolute coefficients per class:")
        for cls in coefs.index:
            top = coefs.loc[cls].drop("intercept").abs().sort_values(
                ascending=False).head(10)
            lines.append(f"  {cls}: " + ", ".join(
                f"{name} ({coefs.loc[cls, name]:+.3f})" for name in top.index))

    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_,
                        index=feature_names).sort_values(ascending=False)
        imp.to_csv(os.path.join(out_dir, f"impurity_importances_{model_name}.csv"),
                   header=["importance"])
        lines.append("\ntop 10 impurity-based feature importances:")
        for name, value in imp.head(10).items():
            lines.append(f"  {name}: {value:.4f}")
        lines.append("  (impurity importances are biased toward high-cardinality "
                     "features; prefer the permutation results from step 5)")

    if hasattr(estimator, "theta_"):  # GaussianNB
        lines.append(f"class priors: "
                     f"{dict(zip(class_labels, np.round(estimator.class_prior_, 4)))}")
        lines.append(f"per-class feature means: {estimator.theta_.shape}")
        lines.append(f"variance smoothing applied: {estimator.var_smoothing:g}")

    if hasattr(estimator, "support_vectors_"):  # SVC
        lines.append(f"support vectors: {estimator.n_support_.sum()} of "
                     f"{estimator.shape_fit_[0]} training sequences "
                     f"({dict(zip(class_labels, estimator.n_support_))})")
        lines.append("  a support-vector count close to the sample count means "
                     "the model is leaning on nearly every point, which is "
                     "common with small data and a soft margin")

    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(config.FINAL_DIR, exist_ok=True)

    best = pd.read_pickle(os.path.join(config.OUT_DIR, "best_per_model.pkl"))
    X, y = load_features()
    feature_names = list(X.columns)
    class_labels = sorted(pd.Series(y).unique().tolist())

    print(f"Refitting {len(best)} models on all {len(X)} labeled sequences\n")

    hyperparams, descriptions, fitted = {}, [], {}
    for _, row in best.iterrows():
        model_name, params = row["model"], row["params"]

        build_kwargs = {}
        if model_name == "SVM_RBF":
            build_kwargs["probability"] = config.SVM_PROBABILITY_IN_FINAL

        clf = cv.fit_one(model_name, params, X.values, y, **build_kwargs)
        fitted[model_name] = clf

        path = os.path.join(config.FINAL_DIR, f"model_{model_name}.joblib")
        joblib.dump({"model": clf,
                     "feature_names": feature_names,
                     "classes": class_labels,
                     "params": params}, path)

        hyperparams[model_name] = {
            "params": params,
            "cv_macro_f1_mean": float(row["macro_f1_mean"]),
            "cv_macro_f1_std": float(row["macro_f1_std"]),
            "cv_balanced_accuracy_mean": float(row["balanced_acc_mean"]),
            "cv_accuracy_mean": float(row["accuracy_mean"]),
        }
        descriptions.append(
            describe_fitted(model_name, clf, feature_names, class_labels,
                            config.FINAL_DIR))
        print(f"  {model_name:20s} saved -> {os.path.basename(path)}")

    with open(os.path.join(config.FINAL_DIR, "hyperparameters.json"), "w") as fh:
        json.dump(hyperparams, fh, indent=2, default=str)
    with open(os.path.join(config.FINAL_DIR, "learned_parameters.txt"), "w") as fh:
        fh.write("\n".join(descriptions))

    # --- predict new sequences ---------------------------------------------
    if not os.path.exists(config.NEW_SEQ_CSV_PATH):
        print(f"\nNo file at {config.NEW_SEQ_CSV_PATH}; skipping prediction. "
              f"The fitted models are saved and ready whenever you have one.")
        return

    print(f"\nPredicting new sequences")
    new_df = features.load_new_sequences()
    pca_table = pd.read_csv(
        os.path.join(config.FEATURE_DIR, "pca_table.csv"), index_col=0)

    X_new = features.build_feature_matrix(new_df, pca_table)
    missing = set(feature_names) - set(X_new.columns)
    if missing:
        raise ValueError(f"New-sequence features are missing columns: {sorted(missing)}")
    X_new = X_new[feature_names]   # identical column order to training

    predictions = new_df[["sequence"]].copy()
    for model_name, clf in fitted.items():
        predictions[f"{model_name}_pred"] = clf.predict(X_new.values)

    pred_cols = [c for c in predictions.columns if c.endswith("_pred")]
    votes = predictions[pred_cols]
    predictions["consensus_pred"] = votes.mode(axis=1)[0]
    predictions["consensus_agreement"] = votes.apply(
        lambda r: (r == r.mode()[0]).sum(), axis=1).astype(str) + f"/{len(pred_cols)}"

    out_path = os.path.join(config.FINAL_DIR, "new_sequence_predictions.csv")
    predictions.to_csv(out_path, index=False)

    print(f"  {len(predictions)} sequences predicted -> {out_path}")
    print(f"  Consensus label counts: "
          f"{predictions['consensus_pred'].value_counts().to_dict()}")
    unanimous = (predictions["consensus_agreement"]
                 == f"{len(pred_cols)}/{len(pred_cols)}").sum()
    print(f"  All {len(pred_cols)} models agreed on {unanimous}/{len(predictions)} "
          f"sequences -- treat split votes as low-confidence calls")


if __name__ == "__main__":
    main()