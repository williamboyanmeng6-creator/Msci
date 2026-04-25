"""
morgan_rf_dataset_generation.py

Generates reduced-dimensionality Morgan fingerprint datasets by selecting
the most informative bits using Random Forest feature importance.

Background:
    I initially tried cumulative importance-mass thresholding (keep bits until
    X% of importance is covered), but for binary Morgan fingerprints this turns
    out to be a poor strategy. A tiny number of bits (mostly encoding the
    backbone class) dominate RF importance, so 70% mass is reached at k=2-3
    bits -- way too few to be useful for GP modelling. Instead I switched to
    a percentile cutoff: keep the top (1 - T) fraction of bits by rank.
    At T=0.70 this gives ~1229 bits, T=0.80 ~819 bits, T=0.90 ~410 bits.

    For the TanimotoGP these high-dimensional inputs are fine because the
    kernel's cost scales O(n^3) in the number of *molecules*, not dimensions,
    and we only ever have n <= 115 in the BO budget.

Inputs:
    Dataset/RF_importance_threshold/Ritsuki_RDKit_RF_importanceMass_0.9_k80_withE0.csv
    (the 2625-molecule base dataset with SMILES columns)

Outputs:
    Dataset/RF_importance_threshold/Ritsuki_Morgan_RF_importancePercentile_{T}_k{k}_withE0.csv
    for T in [0.70, 0.80, 0.90]

Run time: roughly 2-5 minutes on a laptop.

Usage:
    python src/data/morgan_rf_dataset_generation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Paths — edit BASE_DIR to point at your local Dataset folder
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2] / "data"
RF_DIR   = BASE_DIR / "RF_importance_threshold"

# Base dataset: 2625 molecules with oxidised_smiles & reduced_smiles columns
BASE_DATASET = RF_DIR / "Ritsuki_RDKit_RF_importanceMass_0.9_k80_withE0.csv"

TARGET    = "E0_vs_SHE_V"
SID       = "structure_id"
META_COLS = [SID, "Backbone", "Class", "Functional Group",
             "oxidised_smiles", "reduced_smiles"]

# Percentile thresholds for feature selection
THRESHOLDS = [0.70, 0.80, 0.90]
FP_RADIUS  = 2        # ECFP4
FP_NBITS   = 2048     # per SMILES -> 4096 combined
RF_TREES   = 500
RF_SEED    = 42
SENTINEL   = -5.0     # drop obvious outliers (unreliable measurements)


# ---------------------------------------------------------------------------
# Helper: SMILES -> Morgan fingerprint matrix
# ---------------------------------------------------------------------------
def smiles_to_fp(smiles_series, radius, n_bits):
    """
    Convert a pandas Series of SMILES strings to a binary fingerprint matrix.
    Any molecule that fails to parse gets an all-zero row (failsafe).
    Returns a float32 array of shape (n_molecules, n_bits).
    """
    rows = []
    for smi in smiles_series:
        mol = None
        if pd.notna(smi):
            mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            rows.append(np.frombuffer(bv.ToBitString().encode(), dtype="uint8") - ord("0"))
        else:
            rows.append(np.zeros(n_bits, dtype=np.uint8))
    return np.vstack(rows).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RF_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Morgan RF Importance-Mass Dataset Generation")
    print("=" * 60)

    # 1. Load and clean the base dataset
    if not BASE_DATASET.exists():
        raise FileNotFoundError(f"Base dataset not found:\n  {BASE_DATASET}")

    df_base = pd.read_csv(BASE_DATASET)
    print(f"\nLoaded: {BASE_DATASET.name}  shape={df_base.shape}")

    y_s  = pd.to_numeric(df_base[TARGET], errors="coerce")
    keep = y_s.notna() & (y_s > SENTINEL)
    df   = df_base.loc[keep].copy().reset_index(drop=True)
    y    = df[TARGET].astype(float).values
    N    = len(df)
    print(f"After sentinel filter (>{SENTINEL} V): {N} rows")

    # Check for duplicate structure_ids (shouldn't matter since we use
    # positional indexing throughout, but worth flagging)
    n_unique = df[SID].nunique()
    if n_unique < N:
        print(f"  Note: {N - n_unique} duplicate structure_ids found -- "
              "using positional alignment, should be fine.")
    else:
        print(f"  structure_ids: all {N} unique")

    # 2. Compute ECFP4 fingerprints for oxidised and reduced forms separately,
    #    then concatenate to get a 4096-dimensional combined representation.
    #    The idea is to capture both redox states in a single feature vector.
    print(f"\nComputing ECFP4 (radius={FP_RADIUS}, {FP_NBITS} bits each)...")
    X_ox  = smiles_to_fp(df["oxidised_smiles"], FP_RADIUS, FP_NBITS)
    X_red = smiles_to_fp(df["reduced_smiles"],  FP_RADIUS, FP_NBITS)
    X_all = np.hstack([X_ox, X_red])
    print(f"  Combined shape: {X_all.shape}")

    ox_names  = [f"ox_FPbit_{i}"  for i in range(FP_NBITS)]
    red_names = [f"red_FPbit_{i}" for i in range(FP_NBITS)]
    all_names = ox_names + red_names

    # 3. Train a Random Forest on all 4096 bits to score their importance.
    #    High training R^2 is expected here -- we're just using RF as a
    #    feature ranking tool, not for prediction itself.
    print(f"\nTraining RF ({RF_TREES} trees)...")
    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=-1)
    rf.fit(X_all, y)
    importances = rf.feature_importances_
    print(f"  Train R2: {rf.score(X_all, y):.4f}")

    sorted_idx  = np.argsort(importances)[::-1]
    cumsum      = np.cumsum(importances[sorted_idx])
    cumsum_norm = cumsum / cumsum[-1]

    # 4. Plot the cumulative importance curve (sanity check)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(cumsum_norm) + 1), cumsum_norm,
            color="#1f77b4", linewidth=1.5, label="Cumulative importance")
    for thr in THRESHOLDS:
        cutoff   = np.quantile(importances, thr)
        k_thr    = int(np.sum(importances > cutoff))
        imp_at_k = float(cumsum_norm[k_thr - 1]) if k_thr > 0 else 0.0
        ax.axvline(k_thr, color="orange", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.axhline(imp_at_k, color="orange", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(k_thr + 20, imp_at_k - 0.02,
                f"top {100*(1-thr):.0f}% -> k={k_thr}", fontsize=8, color="darkorange")
    ax.set_xlabel("Number of top bits (sorted by RF importance)")
    ax.set_ylabel("Cumulative importance fraction")
    ax.set_title("Morgan RF cumulative importance -- 4096-bit combined fingerprint")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RF_DIR / "Morgan_RF_importance_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved importance curve plot.")

    # 5. For each percentile threshold, keep bits above the cutoff and save a CSV.
    meta_df = df[META_COLS + [TARGET]].copy()
    summary_rows = []

    for thr in THRESHOLDS:
        cutoff  = np.quantile(importances, thr)
        mask    = importances > cutoff
        top_idx = np.where(mask)[0]
        top_idx = top_idx[np.argsort(importances[top_idx])[::-1]]  # sort by importance
        k       = len(top_idx)

        top_names = [all_names[i] for i in top_idx]
        imp_frac  = float(importances[top_idx].sum())

        X_sel  = X_all[:, top_idx]
        fp_df  = pd.DataFrame(X_sel, columns=top_names)
        out_df = pd.concat([meta_df.reset_index(drop=True), fp_df], axis=1)

        fname = f"Ritsuki_Morgan_RF_importancePercentile_{thr}_k{k}_withE0.csv"
        out_df.to_csv(RF_DIR / fname, index=False)

        print(f"\n  T={thr} -> {k} bits, covering {imp_frac:.1%} of RF importance")
        print(f"  Saved: {fname}  shape={out_df.shape}")
        summary_rows.append({"threshold": thr, "k": k, "imp_frac": imp_frac,
                              "n_rows": len(out_df), "filename": fname})

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Threshold':>10}  {'k (bits)':>10}  {'Imp. covered':>13}  {'Rows':>6}  File")
    for r in summary_rows:
        print(f"  {r['threshold']:.2f}      {r['k']:>10}  "
              f"{r['imp_frac']:>13.1%}  {r['n_rows']:>6}  {r['filename']}")
    print(f"\nAll files written to: {RF_DIR}")


if __name__ == "__main__":
    main()
