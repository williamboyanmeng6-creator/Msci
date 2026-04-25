"""
rdkit_feature_generation.py

Generates the RDKit descriptor feature sets used in the Axis 2 and Axis 3
experiments (acquisition function and initialisation comparisons).

Three distinct feature engineering approaches are applied to the same 2625-
molecule pool, producing separate CSVs for each:

  1. Importance-mass threshold selection on RDKit descriptors
     Keep the minimum set of features whose cumulative RF importance reaches
     threshold τ. At τ=0.7 this gives ~101 features (k101), τ=0.8 → ~44 (k44),
     τ=0.9 → ~29 (k29). These are the feature sets used in the Axis 2 LogEI
     vs. UCB comparison and the Axis 3 initialisation comparison.

  2. PCA dimensionality reduction (unsupervised)
     Apply PCA to the standardised 200-descriptor RDKit matrix and retain:
       - Fixed 2 components  (PCA 2D — very low dim, good for visualisation)
       - 90% variance explained (~12 components)
       - 95% variance explained (~18 components)
     PCA is unsupervised so doesn't use the E0 labels. RDKit descriptors have
     natural correlations (e.g. MW correlates with atom count) so PCA does a
     reasonable job of compression. Morgan bits don't have the same structure
     so PCA wasn't applied to those.

  3. Fixed-k top selection already done by morgan_rf_dataset_generation.py
     for Morgan fingerprints.

The selected-initialisation pool (12 curated molecules) is also saved here.
The curation strategy: pick worst, median, and best molecule from each of the
four backbone classes (AQ, BQ, PTZ, PHZ) → 4×3 = 12 molecules.
This covers the E0 range evenly while ensuring chemical diversity.

Inputs:
    data/Ritsuki_full_data_with_rdkit_descriptors.csv

Outputs:
    data/RF_importance_threshold/Ritsuki_RDKit_RF_importanceMass_{τ}_k{k}_withE0.csv
    data/Ritsuki_dataset_RDKitcleared_PCA_fixed.csv
    data/Ritsuki_dataset_RDKitcleared_PCA_90.csv
    data/Ritsuki_dataset_RDKitcleared_PCA_95.csv
    data/INIT_POOLS/INIT_SELECTED_worst_median_best_by_class.csv

Usage:
    python src/data/rdkit_feature_generation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2] / "data"
IN_CSV    = BASE_DIR / "Ritsuki_full_data_with_rdkit_descriptors.csv"
RF_DIR    = BASE_DIR / "RF_importance_threshold"
INIT_DIR  = BASE_DIR / "INIT_POOLS"

TARGET_COL = "E0_vs_SHE_V"
SID_COL    = "structure_id"
SENTINEL   = -5.0

# Columns that aren't features
META_COLS  = {SID_COL, "Backbone", "Class", "Functional Group",
              "oxidised_smiles", "reduced_smiles", "random_initial_set"}

# Importance-mass thresholds for RDKit feature selection
THRESHOLDS = [0.7, 0.8, 0.9]

RF_TREES = 500
RF_SEED  = 42

# PCA variance thresholds and fixed-k options
PCA_VARIANCE_LEVELS = [0.90, 0.95]
PCA_FIXED_K         = 2


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
def load_and_clean(csv_path: Path):
    """Load the base dataset, drop rows where RDKit descriptors couldn't be
    computed (these were molecules with valence errors in the original dataset),
    and impute any remaining NaNs with column medians."""
    df = pd.read_csv(csv_path)
    y  = pd.to_numeric(df[TARGET_COL], errors="coerce")
    keep = y.notna() & (y > SENTINEL)
    df = df.loc[keep].copy().reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in META_COLS and c != TARGET_COL]
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    # Drop any columns that are completely empty
    X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]

    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(X_df.values).astype(float)

    y_vals = df[TARGET_COL].astype(float).values
    return df, X, y_vals, list(X_df.columns)


# ---------------------------------------------------------------------------
# RF importance-mass threshold selection
# ---------------------------------------------------------------------------
def importance_mass_selection(X: np.ndarray, y: np.ndarray,
                               feat_names: list, thresholds: list):
    """
    Train a RF regressor and for each threshold τ, find the smallest set of
    features (sorted by importance, descending) whose cumulative importance
    reaches τ. This is the "importance mass" criterion.

    Note: for Morgan bits, this method degenerates badly (a few backbone-encoding
    bits dominate importance, so τ=0.9 is reached at k~7), which is why the
    percentile approach is used for Morgan FPs instead (see
    morgan_rf_dataset_generation.py). For RDKit continuous descriptors the
    importance is more spread out, so this criterion makes more sense.
    """
    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    sorted_imp = importances[sorted_idx]
    cumsum_imp = np.cumsum(sorted_imp)

    results = {}
    for thr in thresholds:
        # First index where cumulative importance >= thr
        k = int(np.searchsorted(cumsum_imp, thr)) + 1
        top_idx   = sorted_idx[:k]
        top_names = [feat_names[i] for i in top_idx]
        results[thr] = {"k": k, "top_idx": top_idx, "top_names": top_names,
                        "importances": importances}

    return results, rf


def zscore(X: np.ndarray) -> tuple:
    """Z-score columns. Returns (X_scaled, mu, sd) — mu/sd can be used to
    scale new data the same way."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# PCA feature generation
# ---------------------------------------------------------------------------
def generate_pca_features(X: np.ndarray, df: pd.DataFrame,
                           feat_names: list):
    """
    Fit PCA on the standardised feature matrix and save three versions:
    - PCA_fixed: 2 components (lowest meaningful dimension)
    - PCA_90: enough components to explain 90% of variance
    - PCA_95: enough components to explain 95% of variance
    """
    X_z, _, _ = zscore(X)
    pca_full   = PCA()
    pca_full.fit(X_z)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    meta_df = df[[SID_COL, "Backbone", "Class", "Functional Group",
                  "oxidised_smiles", "reduced_smiles", TARGET_COL]].copy()

    outputs = {}

    # Fixed 2-component PCA
    pca_2 = PCA(n_components=PCA_FIXED_K)
    X_pca2 = pca_2.fit_transform(X_z)
    cols_2 = [f"PC{i+1}" for i in range(PCA_FIXED_K)]
    out_df = pd.concat([meta_df.reset_index(drop=True),
                        pd.DataFrame(X_pca2, columns=cols_2)], axis=1)
    outputs["fixed"] = (out_df, PCA_FIXED_K,
                        float(pca_2.explained_variance_ratio_.sum()))

    # Variance-threshold PCA
    for var_thr in PCA_VARIANCE_LEVELS:
        n_comp = int(np.searchsorted(cumvar, var_thr)) + 1
        pca_v  = PCA(n_components=n_comp)
        X_pcav = pca_v.fit_transform(X_z)
        cols_v = [f"PC{i+1}" for i in range(n_comp)]
        out_df = pd.concat([meta_df.reset_index(drop=True),
                            pd.DataFrame(X_pcav, columns=cols_v)], axis=1)
        thr_pct = int(var_thr * 100)
        outputs[f"{thr_pct}"] = (out_df, n_comp,
                                  float(pca_v.explained_variance_ratio_.sum()))

    return outputs, cumvar


# ---------------------------------------------------------------------------
# Selected-init pool generation
# ---------------------------------------------------------------------------
def generate_selected_init(df: pd.DataFrame, y: np.ndarray):
    """
    Build the 12-molecule curated initialisation set used in Axis 3.

    Strategy: pick worst, median, and best E0 molecule from each backbone
    class (AQ, BQ, PTZ, PHZ). This gives 4 × 3 = 12 molecules that:
    - span the full range of the target distribution
    - represent all four chemical families in the dataset
    - are fully deterministic (same 12 molecules for every seed)

    The selected init was consistently outperformed by the random hard-init
    in my experiments, likely because the hard-init still draws from a much
    wider region of chemical space by avoiding only the top performers.
    """
    df = df.copy()
    df["_y"] = y

    rows = []
    for cls in df["Class"].dropna().unique():
        sub = df[df["Class"] == cls].sort_values("_y")
        n   = len(sub)
        if n == 0:
            continue
        picks = {
            "worst":  sub.iloc[0],
            "median": sub.iloc[n // 2],
            "best":   sub.iloc[-1],
        }
        for pick_name, row in picks.items():
            rows.append({
                "Class":          cls,
                "pick":           pick_name,
                SID_COL:          row[SID_COL],
                TARGET_COL:       row["_y"],
                "class_median_E0": float(sub["_y"].median()),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RF_DIR.mkdir(parents=True, exist_ok=True)
    INIT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RDKit Feature Generation")
    print("=" * 60)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found:\n  {IN_CSV}")

    df, X, y, feat_names = load_and_clean(IN_CSV)
    N, D = X.shape
    print(f"\nLoaded dataset: {N} rows, {D} RDKit features")
    print(f"E0 range: [{y.min():.3f}, {y.max():.3f}] V")

    # 1. RF importance-mass selection
    print(f"\nTraining RF ({RF_TREES} trees) for importance-mass selection...")
    sel_results, rf = importance_mass_selection(X, y, feat_names, THRESHOLDS)

    meta_df = df[[SID_COL, "Backbone", "Class", "Functional Group",
                  "oxidised_smiles", "reduced_smiles", TARGET_COL]].copy()

    for thr, res in sel_results.items():
        k       = res["k"]
        top_idx = res["top_idx"]
        X_sel, mu, sd = zscore(X[:, top_idx])

        cols  = [feat_names[i] for i in top_idx]
        out_df = pd.concat([meta_df.reset_index(drop=True),
                            pd.DataFrame(X_sel, columns=cols)], axis=1)
        fname = f"Ritsuki_RDKit_RF_importanceMass_{thr}_k{k}_withE0.csv"
        out_df.to_csv(RF_DIR / fname, index=False)
        print(f"  τ={thr} -> k={k} features -> {fname}")

    # Plot cumulative importance curve
    importances = sel_results[THRESHOLDS[0]]["importances"]
    sorted_imp  = np.sort(importances)[::-1]
    cumsum      = np.cumsum(sorted_imp) / sorted_imp.sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(cumsum) + 1), cumsum,
            color="#1f77b4", linewidth=1.5)
    for thr, res in sel_results.items():
        ax.axvline(res["k"], color="orange", linestyle=":", linewidth=0.8, alpha=0.8)
        ax.text(res["k"] + 1, thr - 0.02, f"τ={thr}, k={res['k']}",
                fontsize=8, color="darkorange")
    ax.set_xlabel("Number of top features (sorted by RF importance)")
    ax.set_ylabel("Cumulative importance fraction")
    ax.set_title("RDKit RF cumulative importance")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RF_DIR / "RDKit_RF_importance_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. PCA features
    print("\nGenerating PCA feature sets...")
    pca_outputs, cumvar = generate_pca_features(X, df, feat_names)
    for key, (out_df, n_comp, var_exp) in pca_outputs.items():
        fname = f"Ritsuki_dataset_RDKitcleared_PCA_{key}.csv"
        out_df.to_csv(BASE_DIR / fname, index=False)
        print(f"  PCA {key}: {n_comp} components, {var_exp:.1%} variance -> {fname}")

    # PCA scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, "o-", ms=3, linewidth=1)
    for v in PCA_VARIANCE_LEVELS:
        ax.axhline(v, color="orange", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(len(cumvar) * 0.7, v + 0.01, f"{int(v*100)}%", fontsize=9,
                color="darkorange")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("PCA scree plot — RDKit descriptors")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "RDKit_PCA_scree.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Selected-init pool
    print("\nGenerating selected-init pool (worst/median/best per class)...")
    init_df = generate_selected_init(df, y)
    init_path = INIT_DIR / "INIT_SELECTED_worst_median_best_by_class.csv"
    init_df.to_csv(init_path, index=False)
    print(f"  Saved {len(init_df)} molecules -> {init_path.name}")
    print(init_df[["Class", "pick", SID_COL, TARGET_COL]].to_string(index=False))

    print(f"\nAll outputs written to: {BASE_DIR}")


if __name__ == "__main__":
    main()
