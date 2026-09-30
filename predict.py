#!/usr/bin/env python3
"""
predict.py -- classify tetrapeptide sequences with the trained models.

Uses the fitted models shipped in results_4block_pca/final/ directly. No
retraining, no hyperparameter search output, and no `aaindex` package needed.

Command line:

    python predict.py GAAC GKLV
    python predict.py --input my_sequences.csv --output predictions.csv
    python predict.py GAAC --model SVM_RBF

From Python or a notebook:

    from predict import predict
    predict(["GAAC", "GKLV"])

Input sequences must be 4 residues long (anchor + A2 + A3 + A4), using the 20
standard amino acids. Position 1 is the fixed "G" anchor and is not used as a
feature; A2-A4 are encoded with the same 4-block AAindex PCA lookup table the
models were trained on.

To use models you trained yourself with the pipeline (steps 01-06), point
--models-dir at that run's final/ directory and --pca-table at its
features/pca_table.csv.
"""

import argparse
import glob
import json
import os
import sys

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(HERE, "results_4block_pca", "final")
DEFAULT_PCA_TABLE = os.path.join(HERE, "results_4block_pca", "features",
                                 "pca_table.csv")

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = ["A2", "A3", "A4"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_models(models_dir=DEFAULT_MODELS_DIR, names=None):
    """Load every model_<name>.joblib in models_dir, or only those in `names`."""
    paths = sorted(glob.glob(os.path.join(models_dir, "model_*.joblib")))
    if not paths:
        raise FileNotFoundError(f"No model_*.joblib files found in {models_dir}")

    # hyperparameters.json lists models best cross-validation score first;
    # keep that order so output columns match 06_final_train_predict.py.
    ranking_path = os.path.join(models_dir, "hyperparameters.json")
    if os.path.exists(ranking_path):
        with open(ranking_path) as fh:
            rank = {name: i for i, name in enumerate(json.load(fh))}
        paths.sort(key=lambda p: rank.get(os.path.basename(p)[6:-7], len(rank)))

    bundles = {}
    for path in paths:
        name = os.path.basename(path)[len("model_"):-len(".joblib")]
        if names is None or name in names:
            bundles[name] = joblib.load(path)

    if names is not None:
        missing = set(names) - set(bundles)
        if missing:
            available = [os.path.basename(p)[6:-7] for p in paths]
            raise ValueError(f"Unknown model(s) {sorted(missing)}. "
                             f"Available: {available}")
    return bundles


def load_pca_table(path=DEFAULT_PCA_TABLE):
    return pd.read_csv(path, index_col=0)


def read_sequence_file(path):
    """Read sequences from a CSV/TXT file.

    Accepts either a file with a 'sequence' column header, or a headerless
    file whose first column holds the sequences (the format of
    new_predict.csv). Any other columns are ignored.
    """
    df = pd.read_csv(path, header=None, dtype=str)
    first = df.iloc[0, 0].strip().lower()
    if first == "sequence":
        df = pd.read_csv(path, dtype=str)
        return df["sequence"].tolist()
    return df.iloc[:, 0].tolist()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def clean_sequences(sequences):
    """Upper-case, strip, and validate. Raises ValueError on bad input."""
    seqs = [str(s).strip().upper() for s in sequences]
    seqs = [s for s in seqs if s]

    problems = []
    for s in seqs:
        if len(s) != 4:
            problems.append(f"{s!r}: expected 4 residues, got {len(s)}")
        elif any(c not in AA_ORDER for c in s[1:]):
            problems.append(f"{s!r}: non-standard amino acid at positions 2-4")
    if problems:
        raise ValueError("Invalid sequence(s):\n  " + "\n  ".join(problems))

    non_g = [s for s in seqs if s[0] != "G"]
    if non_g:
        print(f"Warning: {len(non_g)} sequence(s) do not start with 'G'. The "
              f"models were trained on G-anchored sequences and position 1 is "
              f"ignored, so treat these predictions with care: {non_g}",
              file=sys.stderr)
    return seqs


def encode(sequences, pca_table):
    """Turn sequences into the A2/A3/A4 x PCA-component feature matrix."""
    frames = []
    for i, pos in enumerate(POSITIONS, start=1):
        residues = [s[i] for s in sequences]
        looked_up = pca_table.loc[residues].reset_index(drop=True)
        looked_up.columns = [f"{pos}_{c}" for c in looked_up.columns]
        frames.append(looked_up)
    return pd.concat(frames, axis=1).astype(float)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(sequences, models_dir=DEFAULT_MODELS_DIR,
            pca_table_path=DEFAULT_PCA_TABLE, models=None):
    """Predict the class of each sequence with every loaded model.

    Returns a DataFrame with one row per sequence: each model's predicted
    label, a majority-vote consensus, and how many models agreed with it.
    """
    seqs = clean_sequences(sequences)
    if not seqs:
        raise ValueError("No sequences given.")

    bundles = load_models(models_dir, models)
    X = encode(seqs, load_pca_table(pca_table_path))

    out = pd.DataFrame({"sequence": seqs})
    for name, bundle in bundles.items():
        feature_names = bundle["feature_names"]
        missing = set(feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"PCA table does not match model '{name}'; "
                             f"missing features {sorted(missing)[:5]}...")
        X_model = X[feature_names].values   # identical column order to training
        out[f"{name}_pred"] = bundle["model"].predict(X_model)

    pred_cols = [c for c in out.columns if c.endswith("_pred")]
    votes = out[pred_cols]
    out["consensus_pred"] = votes.mode(axis=1)[0]
    out["consensus_agreement"] = (
        votes.eq(out["consensus_pred"], axis=0).sum(axis=1).astype(str)
        + f"/{len(pred_cols)}"
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Classify G-X-X-X tetrapeptides with the trained models.")
    ap.add_argument("sequences", nargs="*",
                    help="one or more 4-residue sequences, e.g. GAAC GKLV")
    ap.add_argument("-i", "--input",
                    help="CSV/TXT file of sequences (one per line, or a "
                         "'sequence' column)")
    ap.add_argument("-o", "--output",
                    help="write predictions to this CSV instead of only printing")
    ap.add_argument("-m", "--model", action="append", dest="models",
                    help="use only this model (repeatable). Default: all. "
                         "Choices: SVM_RBF, ElasticNetLogistic, RandomForest")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                    help="directory holding model_<name>.joblib files")
    ap.add_argument("--pca-table", default=DEFAULT_PCA_TABLE,
                    help="AAindex PCA lookup table the models were trained with")
    args = ap.parse_args()

    sequences = list(args.sequences)
    if args.input:
        sequences += read_sequence_file(args.input)
    if not sequences:
        ap.error("give sequences on the command line or with --input")

    try:
        result = predict(sequences, args.models_dir, args.pca_table, args.models)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(result.to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
