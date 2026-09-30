#!/usr/bin/env python3
"""
07_export_for_origin.py -- export the data behind Supp. Figs 42 and 43 as CSV
files that Origin can import directly. Naive Bayes is left out everywhere;
edit MODELS below to change which classifiers are exported.

Run from the 3AA/ directory:
    python3 07_export_for_origin.py
    python3 07_export_for_origin.py --results results_4block_pca --out origin_export

Fig 42a needs the per-block PCA explained-variance ratios. If meta.json holds
them they are read from there; otherwise the four PCAs are recomputed from
AAindex1 (needs the `aaindex` package) and checked against pca_table.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

MODELS = ["SVM_RBF", "ElasticNetLogistic", "RandomForest"]
BLOCKS = ["nonsec", "helix", "beta", "othersec"]
BLOCK_ALIASES = {
    "nonsec":   ["nonsec", "non_sec", "nonsecondary", "non-sec"],
    "helix":    ["helix"],
    "beta":     ["beta", "sheet", "strand"],
    "othersec": ["othersec", "other_sec", "other"],
}
N_PCS_PLOT = 20
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def walk(obj, path=()):
    """Yield (path, value) for every leaf list in a nested JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        yield path, obj


def is_numeric_list(v):
    return (len(v) >= 2 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def is_string_list(v):
    return len(v) >= 1 and all(isinstance(x, str) for x in v)


def block_of(path):
    """Which block a JSON path refers to, matching the last key first."""
    for key in reversed(path):
        k = key.lower()
        for block in BLOCKS:
            if k in BLOCK_ALIASES[block]:
                return block
    return None


# --------------------------------------------------------------------------
# Fig 42a -- PCA explained variance per descriptor block
# --------------------------------------------------------------------------
def variance_from_meta(feat_dir):
    meta_path = feat_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    found = {}
    for path, val in walk(meta):
        joined = "/".join(path).lower()
        if ("explained" in joined or "variance" in joined) and is_numeric_list(val):
            b = block_of(path)
            if b and b not in found:
                found[b] = np.asarray(val, float)
    if set(found) == set(BLOCKS):
        print("  explained variance read from meta.json")
        return found
    return None


def variance_recomputed(feat_dir):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    try:
        from aaindex import aaindex1
    except ImportError:
        sys.exit("meta.json has no explained variance and the `aaindex` package "
                 "is not installed. Run: pip install aaindex")

    groups = json.loads((feat_dir / "descriptor_groups.json").read_text())
    codes = {}
    for path, val in walk(groups):
        if is_string_list(val):
            b = block_of(path)
            if b and b not in codes:
                codes[b] = val
    missing = set(BLOCKS) - set(codes)
    if missing:
        sys.exit(f"Could not find descriptor lists for {sorted(missing)} in "
                 f"descriptor_groups.json. Send me its top-level structure.")

    pca_table = pd.read_csv(feat_dir / "pca_table.csv", index_col=0)
    aa_order = [a for a in pca_table.index.astype(str) if a in AA20]
    if len(aa_order) != 20:
        aa_order = AA20

    out = {}
    for b in BLOCKS:
        cols = {}
        for c in codes[b]:
            vals = aaindex1[c]["values"]
            cols[c] = [vals.get(a, np.nan) for a in aa_order]
        M = pd.DataFrame(cols, index=aa_order).apply(pd.to_numeric, errors="coerce")
        M = M.fillna(M.mean())
        M = M.loc[:, M.std(ddof=0) > 0]
        Z = StandardScaler().fit_transform(M.values)
        pca = PCA().fit(Z)
        out[b] = pca.explained_variance_ratio_

        # sanity check: recomputed scores should match pca_table.csv up to sign
        scores = pca.transform(Z)
        corrs = []
        for k in range(scores.shape[1]):
            col = f"{b}_PC{k + 1}"
            if col in pca_table.columns:
                ref = pca_table.loc[aa_order, col].values
                corrs.append(abs(np.corrcoef(ref, scores[:, k])[0, 1]))
        worst = min(corrs) if corrs else float("nan")
        flag = "OK" if worst > 0.99 else "MISMATCH -- check preprocessing in features.py"
        print(f"  {b:9s} {len(codes[b]):4d} descriptors, "
              f"min |r| vs pca_table over {len(corrs)} PCs = {worst:.4f}  [{flag}]")
    return out


def export_pca(feat_dir, out_dir):
    print("Fig 42a: PCA explained variance")
    evr = variance_from_meta(feat_dir) or variance_recomputed(feat_dir)
    wide = pd.DataFrame({"PC": np.arange(1, N_PCS_PLOT + 1)})
    for b in BLOCKS:
        r = np.asarray(evr[b], float)[:N_PCS_PLOT]
        r = np.pad(r, (0, N_PCS_PLOT - len(r)), constant_values=np.nan)
        df = pd.DataFrame({"PC": np.arange(1, N_PCS_PLOT + 1),
                           "individual": r,
                           "cumulative": np.nancumsum(r)})
        df.to_csv(out_dir / f"fig42a_pca_{b}.csv", index=False, float_format="%.6f")
        wide[f"{b}_individual"] = df["individual"]
        wide[f"{b}_cumulative"] = df["cumulative"]
    wide.to_csv(out_dir / "fig42a_pca_all_blocks.csv", index=False, float_format="%.6f")


# --------------------------------------------------------------------------
# Fig 42b/c -- confusion matrices (counts and row-normalized)
# --------------------------------------------------------------------------
def export_confusion(final_dir, feat_dir, out_dir):
    print("Fig 42b/c: confusion matrices")
    y = pd.read_csv(feat_dir / "y.csv").iloc[:, 0].astype(str)
    labels = sorted(y.unique())          # MS, droplet, fiber
    n_seq = len(y)

    summary, long_rows = [], []
    for m in MODELS:
        df = pd.read_csv(final_dir / f"cv_predictions_{m}.csv")
        yt, yp = df["true"].astype(str), df["predicted"].astype(str)
        cm = confusion_matrix(yt, yp, labels=labels)
        rs = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, rs, out=np.zeros(cm.shape), where=rs > 0)

        idx = pd.Index(labels, name="true\\predicted")
        pd.DataFrame(cm, index=idx, columns=labels).to_csv(
            out_dir / f"fig42b_confusion_counts_{m}.csv")
        pd.DataFrame(cmn, index=idx, columns=labels).to_csv(
            out_dir / f"fig42c_confusion_normalized_{m}.csv", float_format="%.4f")

        for i, t in enumerate(labels):
            for j, p in enumerate(labels):
                long_rows.append({"model": m, "true": t, "predicted": p,
                                  "count": int(cm[i, j]),
                                  "fraction_of_true": round(float(cmn[i, j]), 4)})

        acc = accuracy_score(yt, yp)
        bal = balanced_accuracy_score(yt, yp)
        mf1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
        summary.append({"model": m, "n_predictions": len(df), "n_sequences": n_seq,
                        "n_repeats": len(df) / n_seq, "accuracy": acc,
                        "balanced_accuracy": bal, "macro_f1": mf1})
        print(f"  {m:20s} acc {acc:.3f} | balanced {bal:.3f} | macro-F1 {mf1:.3f}")

    pd.DataFrame(long_rows).to_csv(out_dir / "fig42bc_confusion_long_all_models.csv",
                                   index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fig42bc_metrics_summary.csv",
                                 index=False, float_format="%.4f")


# --------------------------------------------------------------------------
# Fig 43 -- block ablation
# --------------------------------------------------------------------------
def block_type(name):
    if " x " in name:
        return "position x category"
    if name.startswith("position:"):
        return "position"
    if name.startswith("category:"):
        return "category"
    return "other"


def export_ablation(final_dir, out_dir):
    print("Fig 43: block ablation")
    df = pd.read_csv(final_dir / "block_ablation.csv")
    df = df[df["model"].isin(MODELS)]
    df = df[~df["block"].astype(str).str.startswith("(none")].copy()
    df["block_type"] = df["block"].map(block_type)

    cols = ["block", "block_type", "n_features_dropped", "delta_vs_baseline",
            "macro_f1_mean", "macro_f1_std"]
    for m in MODELS:
        # most negative first: in an Origin horizontal bar chart row 1 is drawn
        # at the bottom, which reproduces the layout of the Python figure
        d = df[df["model"] == m].sort_values("delta_vs_baseline")[cols]
        d.to_csv(out_dir / f"fig43_block_ablation_{m}.csv", index=False,
                 float_format="%.6f")
        print(f"  {m:20s} {len(d)} blocks")

    wide = df.pivot(index="block", columns="model", values="delta_vs_baseline")[MODELS]
    std = df.pivot(index="block", columns="model", values="macro_f1_std")[MODELS]
    std.columns = [f"{c}_macro_f1_std" for c in std.columns]
    wide = wide.join(std)
    wide.insert(0, "block_type", wide.index.map(block_type))
    wide = wide.loc[wide[MODELS].mean(axis=1).sort_values().index]
    wide.to_csv(out_dir / "fig43_block_ablation_all_models.csv", float_format="%.6f")


# --------------------------------------------------------------------------
README = """Data files for Supplementary Figs 42 and 43 (classifiers: {models})

fig42a_pca_<block>.csv            PC, individual explained-variance ratio, cumulative
fig42a_pca_all_blocks.csv         same, all four blocks side by side
fig42b_confusion_counts_<m>.csv   3x3 counts; rows = true class, columns = predicted
fig42c_confusion_normalized_<m>.csv  same, each row divided by its total (recall)
fig42bc_confusion_long_all_models.csv  one row per cell (model, true, predicted,
                                  count, fraction) -- use as XYZ for Origin heatmaps
fig42bc_metrics_summary.csv       accuracy, balanced accuracy, macro-F1 per model,
                                  computed on the pooled cross-validation predictions
fig43_block_ablation_<m>.csv      change in macro-F1 when each block is removed
                                  (negative = block was helping), sorted most
                                  negative first; macro_f1_std is the fold-to-fold
                                  SD of the ablated model's macro-F1
fig43_block_ablation_all_models.csv  one row per block, one column per model
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_4block_pca")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <results>/origin_export)")
    ap.add_argument("--skip-pca", action="store_true",
                    help="skip Fig 42a (e.g. if aaindex is not installed)")
    # a = ap.parse_args()
    a = ap.parse_args(argv)

    res = Path(a.results)
    feat_dir, final_dir = res / "features", res / "final"
    out_dir = Path(a.out) if a.out else res / "origin_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not a.skip_pca:
        export_pca(feat_dir, out_dir)
    export_confusion(final_dir, feat_dir, out_dir)
    export_ablation(final_dir, out_dir)
    (out_dir / "README.txt").write_text(README.format(models=", ".join(MODELS)))
    print(f"\nWrote {len(list(out_dir.glob('*.csv')))} CSV files to {out_dir}/")


if __name__ == "__main__":
    main(["--results", "/home/hchilukurisa/pikes/projects/LipML/3AA/results_4block_pca"])
