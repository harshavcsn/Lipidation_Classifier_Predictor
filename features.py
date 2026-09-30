"""
features.py -- data loading and feature construction.

Two feature blocks, both keyed on the 3 variable positions A2/A3/A4:

  one-hot        3 x 20  = 60 indicator columns
  AAindex PCA    3 x sum(N_PCS.values()) columns, looked up from a
                 per-amino-acid table built by PCA-ing four separate
                 groups of AAindex descriptors.

The PCA lookup table is derived purely from universal AAindex descriptors,
never from the sequence labels, so it is identical for training sequences and
for brand-new sequences. That is why it can be built once and cached.
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------
def _split_positions(df):
    df["A2"] = df["sequence"].str[1]
    df["A3"] = df["sequence"].str[2]
    df["A4"] = df["sequence"].str[3]
    return df


def _check_sequences(seqs, what):
    bad_len = seqs.str.len() != 4
    if bad_len.any():
        raise ValueError(
            f"Expected 4-character {what} (anchor + A2 + A3 + A4). "
            f"Found {int(bad_len.sum())} bad row(s), e.g. "
            f"{seqs[bad_len].head().tolist()}"
        )

    bad_aa = ~seqs.str[1:].apply(lambda s: all(c in config.AA_ORDER for c in s))
    if bad_aa.any():
        raise ValueError(
            f"Found {what} with characters outside the 20 standard amino "
            f"acids at positions A2-A4, e.g. {seqs[bad_aa].head().tolist()}"
        )

    non_g = seqs.str[0] != "G"
    if non_g.any():
        print(
            f"  !! {int(non_g.sum())} {what} do not start with 'G'. Position 1 "
            f"is discarded regardless, but check these: {seqs[non_g].tolist()}"
        )


def load_labeled_data(csv_path=None):
    csv_path = csv_path or config.CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Labeled data not found at '{csv_path}'. Set CSV_PATH in config.py. "
            f"Expected: no header, two columns 'sequence,label'."
        )

    df = pd.read_csv(csv_path, header=None, names=["sequence", "label"])
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(str).str.strip()
    _check_sequences(df["sequence"], "sequences")
    df = _split_positions(df)

    if config.MERGE_CLASSES:
        df["label"] = df["label"].replace({"MS": "fiber"})
        print("  Merged MS into fiber.")

    print(f"  Loaded {len(df)} labeled sequences from {csv_path}")
    print(f"  Class counts: {df['label'].value_counts().to_dict()}")
    return df


def load_new_sequences(csv_path=None):
    csv_path = csv_path or config.NEW_SEQ_CSV_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"New-sequence file not found at '{csv_path}'. Set NEW_SEQ_CSV_PATH "
            f"in config.py. Expected: no header, a single column of sequences."
        )

    df = pd.read_csv(csv_path, header=None, names=["sequence"])
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    _check_sequences(df["sequence"], "new sequences")
    df = _split_positions(df)
    print(f"  Loaded {len(df)} new sequences from {csv_path}")
    return df


# ---------------------------------------------------------------------------
# AAindex -> 4-category PCA lookup table
# ---------------------------------------------------------------------------
def classify_sec_struct(description):
    """Bucket a sec_struct descriptor into HELIX / BETA / OTHER.

    Order matters: the turn/coil/loop keywords are checked first so that a
    descriptor like "non-helical turn propensity" lands in OTHER rather than
    being caught by the helix keywords.
    """
    desc = description.lower()

    other_kw = ["turn", "coil", "bend", "loop", "linker",
                "non-helical", "non-beta", "non beta"]
    if any(kw in desc for kw in other_kw):
        return "OTHER"

    helix_kw = ["helix", "alpha", "helical"]
    if any(kw in desc for kw in helix_kw):
        return "HELIX"

    beta_kw = ["beta", "sheet", "strand", "extended"]
    if any(kw in desc for kw in beta_kw):
        return "BETA"

    return "OTHER"


def _pca_block(descriptor_ids, clean_descriptors, prefix, n_components,
               seed, scaling):
    """PCA one group of descriptors down to a 20 x n_components table."""
    if len(descriptor_ids) == 0:
        raise ValueError(f"Block '{prefix}' has no descriptors.")

    # rows = amino acids, columns = descriptors in this block
    aa_by_prop = clean_descriptors.loc[descriptor_ids].T

    # PCA on 20 samples can yield at most 19 non-trivial components, and never
    # more components than there are descriptors in the block.
    n_comp = min(n_components, aa_by_prop.shape[1], aa_by_prop.shape[0] - 1)

    x_std = StandardScaler().fit_transform(aa_by_prop.values)
    pca = PCA(n_components=n_comp, random_state=seed)
    scores = pca.fit_transform(x_std)

    if scaling == "minmax":
        scores = MinMaxScaler().fit_transform(scores)
    elif scaling == "zscore":
        scores = StandardScaler().fit_transform(scores)
    elif scaling != "raw":
        raise ValueError(f"Unknown PCA_SCALING '{scaling}'")

    cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    print(f"  [{prefix:9s}] {len(descriptor_ids):4d} descriptors -> "
          f"{n_comp:2d} PCs, cumulative variance {cum_var:.1%}")

    return pd.DataFrame(
        scores,
        index=aa_by_prop.index,
        columns=[f"{prefix}_PC{i + 1}" for i in range(n_comp)],
    )


def build_pca_table(seed=None, scaling=None, n_pcs=None):
    """Build the 20 x N per-amino-acid lookup table from AAindex1.

    Requires the `aaindex` package. Called once by 01_build_features.py; the
    result is cached to CSV so no other script needs the package installed.
    """
    from aaindex import aaindex1

    seed = config.SEED if seed is None else seed
    scaling = config.PCA_SCALING if scaling is None else scaling
    n_pcs = config.N_PCS if n_pcs is None else n_pcs

    aaindex_df = aaindex1.to_dataframe()
    aa_cols = [c for c in aaindex_df.columns if c != "-"]
    clean = aaindex_df[aa_cols].dropna(axis=0, how="any")
    print(f"  AAindex1 descriptors: {aaindex_df.shape[0]} total, "
          f"{clean.shape[0]} with no missing values")

    # --- category 1-3: secondary-structure descriptors, split three ways ----
    sec_records = aaindex1.get_record_by_category("sec_struct")
    sec_records = {k: v for k, v in sec_records.items() if k in clean.index}

    helix_ids, beta_ids, othersec_ids = [], [], []
    for prop_id, record in sec_records.items():
        bucket = classify_sec_struct(record.get("description", ""))
        if bucket == "HELIX":
            helix_ids.append(prop_id)
        elif bucket == "BETA":
            beta_ids.append(prop_id)
        else:
            othersec_ids.append(prop_id)

    # --- category 4: everything that is not tagged sec_struct ---------------
    nonsec_ids = [i for i in clean.index if i not in sec_records]

    groups = {
        "nonsec": sorted(nonsec_ids),
        "helix": sorted(helix_ids),
        "beta": sorted(beta_ids),
        "othersec": sorted(othersec_ids),
    }

    tables = [
        _pca_block(ids, clean, name, n_pcs[name], seed, scaling)
        for name, ids in groups.items()
    ]

    pca_table = pd.concat(tables, axis=1).reindex(config.AA_ORDER)
    if pca_table.isna().any().any():
        raise ValueError("PCA lookup table has NaNs -- an amino acid is missing.")

    print(f"  Combined lookup table: {pca_table.shape[0]} amino acids x "
          f"{pca_table.shape[1]} components")
    return pca_table, groups


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def one_hot_encode(df):
    frames = []
    for pos in config.POSITIONS:
        cat = pd.Categorical(df[pos], categories=config.AA_ORDER)
        oh = pd.get_dummies(cat, prefix=pos, dtype=int)
        oh.index = df.index
        frames.append(oh)
    return pd.concat(frames, axis=1)


def pca_lookup_encode(df, pca_table):
    frames = []
    for pos in config.POSITIONS:
        looked_up = pca_table.loc[df[pos].values].reset_index(drop=True)
        looked_up.columns = [f"{pos}_{c}" for c in looked_up.columns]
        frames.append(looked_up)
    out = pd.concat(frames, axis=1)
    out.index = df.index
    return out


def build_feature_matrix(df, pca_table):
    blocks = []
    if config.USE_ONEHOT:
        blocks.append(one_hot_encode(df).reset_index(drop=True))
    if config.USE_PCA:
        blocks.append(pca_lookup_encode(df, pca_table).reset_index(drop=True))
    if not blocks:
        raise ValueError("Both USE_ONEHOT and USE_PCA are False -- no features.")
    return pd.concat(blocks, axis=1).astype(float)


def feature_group(column):
    """Map a feature name like 'A2_helix_PC3' to ('A2', 'helix').

    One-hot columns look like 'A2_K' and map to ('A2', 'onehot').
    """
    parts = column.split("_")
    position = parts[0]
    if len(parts) == 2:
        return position, "onehot"
    return position, parts[1]
