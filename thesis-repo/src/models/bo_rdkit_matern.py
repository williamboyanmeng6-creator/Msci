"""
bo_rdkit_matern.py

Axis 1 experiment — RDKit physicochemical descriptors with Matérn-2.5 ARD kernel.

This is the first representation tested in the systematic comparison. RDKit
descriptors are continuous (logP, TPSA, MW, ring counts, etc.) and after
z-scoring they work well with the standard Matérn-2.5 kernel, which assumes
a Euclidean distance structure. Top-k descriptors are selected by RF importance
to reduce dimensionality from ~200 to 25, 50, or 100 features.

Also covers the PCA variants (PCA 2D, PCA 90%, PCA 95%) since the BO loop
is identical — only the feature matrix changes.

Experiment settings (kept identical across all Axis 1 runs):
    - Surrogate     : SingleTaskGP with Matérn-2.5 ARD (BoTorch default)
    - Acquisition   : LogExpectedImprovement
    - Initialisation: selected-init — fixed 12 molecules (worst/median/best
                      per backbone class), same set across all 10 seeds
    - Budget        : 100 evaluations total
    - Seeds         : 100-109

The selected-init is deterministic (zero inter-seed variance), which is useful
for isolating the representation effect from initialisation noise in Axis 1.
Axis 3 revisits initialisation strategy specifically.

Inputs:
    data/RF_importance_threshold/Ritsuki_RDKit_RF_importanceMass_*_withE0.csv
    data/Ritsuki_dataset_RDKitcleared_PCA_*.csv
    data/INIT_POOLS/INIT_SELECTED_worst_median_best_by_class.csv

Outputs:
    data/BO_multiseed_RDKIT/<dataset_name>/offline_BO_trace_seed*.csv
    data/BO_multiseed_RDKIT/<dataset_name>/MS_summary_BO_seeds.csv

Usage:
    python src/models/bo_rdkit_matern.py
    # To run only RDKit top-k (skip PCA), set INCLUDE_PCA = False
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[2] / "data"
RF_DIR     = BASE_DIR / "RF_importance_threshold"
INIT_FILE  = BASE_DIR / "INIT_POOLS" / "INIT_SELECTED_worst_median_best_by_class.csv"
OUT_ROOT   = BASE_DIR / "BO_multiseed_RDKIT"

TARGET_COL = "E0_vs_SHE_V"
SID_COL    = "structure_id"
SENTINEL   = -5.0

SEEDS       = list(range(100, 110))
N_INIT      = 12
BUDGET      = 100
THRESH      = 1.30
GP_MAXITER  = 50
DEVICE      = "cpu"
DTYPE       = torch.double

INCLUDE_PCA = True   # set False to skip the PCA datasets

EXCLUDE = {TARGET_COL, SID_COL, "Backbone", "Class", "Functional Group",
           "oxidised_smiles", "reduced_smiles", "random_initial_set"}

# Datasets to run (glob pattern for RF-selected, plus PCA CSVs if INCLUDE_PCA)
RDKIT_GLOB = "Ritsuki_RDKit_RF_importanceMass_*_withE0.csv"
PCA_CSVS   = [
    BASE_DIR / "Ritsuki_dataset_RDKitcleared_PCA_fixed.csv",
    BASE_DIR / "Ritsuki_dataset_RDKitcleared_PCA_90.csv",
    BASE_DIR / "Ritsuki_dataset_RDKitcleared_PCA_95.csv",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def best_so_far(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)


def time_to_threshold(curve: np.ndarray, thresh: float):
    idx = np.where(curve >= thresh)[0]
    return int(idx[0] + 1) if len(idx) else np.nan


def zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def fit_matern_gp(X_train: torch.Tensor, y_train: torch.Tensor) -> SingleTaskGP:
    """Standard SingleTaskGP with Matérn-2.5 ARD (BoTorch default kernel)."""
    y_std = standardize(y_train)
    model = SingleTaskGP(X_train, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


# ---------------------------------------------------------------------------
# Selected-init lookup
# ---------------------------------------------------------------------------
def load_selected_init(df: pd.DataFrame) -> list:
    """
    Look up the 12 curated molecules in the current dataset by structure_id.
    These are the same 12 across all seeds (worst/median/best per backbone class).
    If any structure_id from the pool is missing from the dataset, raise an error.
    """
    if not INIT_FILE.exists():
        raise FileNotFoundError(
            f"Selected-init pool not found: {INIT_FILE}\n"
            "Run src/data/rdkit_feature_generation.py first."
        )
    init_df = pd.read_csv(INIT_FILE)
    sids    = init_df[SID_COL].tolist()

    sid_to_idx = {sid: i for i, sid in enumerate(df[SID_COL])}
    indices = []
    for sid in sids:
        if sid not in sid_to_idx:
            raise KeyError(f"Selected-init molecule '{sid}' not found in dataset.")
        indices.append(sid_to_idx[sid])
    return indices


# ---------------------------------------------------------------------------
# BO loop (Matérn / LogEI)
# ---------------------------------------------------------------------------
def run_bo_loop(X_t: torch.Tensor, y_t: torch.Tensor, y: np.ndarray,
                init_idx: list, n_iter: int):
    """
    BO loop with Matérn GP + LogEI acquisition.
    X must be z-scored before calling this (Matérn uses Euclidean distances).
    """
    remaining  = [i for i in range(len(y)) if i not in set(init_idx)]
    X_obs      = X_t[init_idx]
    y_obs      = y_t[init_idx]
    y_chosen   = y[init_idx].tolist()
    best_curve = best_so_far(np.array(y_chosen)).astype(float)

    for _ in range(n_iter):
        model  = fit_matern_gp(X_obs, y_obs)
        best_f = float(standardize(y_obs).max().item())
        acq    = LogExpectedImprovement(model=model, best_f=best_f)

        X_pool = X_t[remaining]
        with torch.no_grad():
            vals = acq(X_pool.unsqueeze(1)).squeeze(-1)

        pick_pos = int(torch.argmax(vals).item())
        nxt      = remaining[pick_pos]

        y_chosen.append(float(y_t[nxt].item()))
        X_obs    = torch.cat([X_obs, X_t[nxt].view(1, -1)], dim=0)
        y_obs    = torch.cat([y_obs, y_t[nxt].view(1, 1)],  dim=0)
        remaining.pop(pick_pos)
        best_curve = np.append(best_curve, max(best_curve[-1], y_chosen[-1]))

    return best_curve, np.array(y_chosen)


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------
def run_dataset(csv_path: Path):
    name    = csv_path.stem
    out_dir = OUT_ROOT / name
    ensure_dir(out_dir)

    df   = pd.read_csv(csv_path)
    y_s  = pd.to_numeric(df[TARGET_COL], errors="coerce")
    mask = y_s.notna() & (y_s > SENTINEL)
    df   = df.loc[mask].copy().reset_index(drop=True)
    y    = df[TARGET_COL].astype(float).values
    N    = len(df)
    B    = min(BUDGET, N)
    n_iter = B - N_INIT

    # Build feature matrix: strip meta/target, impute, z-score
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]
    imp  = SimpleImputer(strategy="median")
    X    = imp.fit_transform(X_df.values).astype(float)
    X    = zscore_cols(X)

    X_t  = torch.tensor(X, device=DEVICE, dtype=DTYPE)
    y_t  = torch.tensor(y, device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    # The selected-init uses the same 12 molecules for every seed
    sel_init = load_selected_init(df)

    d = X_t.shape[1]
    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"  N={N}  d={d}  B={B}  n_iter={n_iter}")

    all_best = []
    all_ttf  = []
    summary_rows = []

    for seed in SEEDS:
        print(f"  [seed {seed}] Matérn LogEI (selected-init)...", end=" ", flush=True)
        curve, y_obs = run_bo_loop(X_t, y_t, y, sel_init, n_iter)
        ttf = time_to_threshold(curve, THRESH)
        print(f"TTF={ttf}  final={curve[-1]:.4f}")

        # Save full trace (molecule metadata + BO results)
        trace_df = df.iloc[sel_init + [i for i in range(N)
                                       if i not in set(sel_init)][:n_iter]].copy()
        trace_df = df.copy()
        trace_df["bo_order"]    = np.arange(1, N + 1)
        trace_df["y_observed"]  = np.concatenate([y_obs, y[~np.isin(np.arange(N), sel_init)]])
        trace_df["best_so_far"] = curve[:N] if len(curve) >= N else np.pad(curve, (0, N - len(curve)), constant_values=curve[-1])

        pd.DataFrame({
            "y_observed":  y_obs,
            "best_so_far": curve,
        }).to_csv(out_dir / f"offline_BO_trace_seed{seed}.csv", index=False)

        all_best.append(curve)
        all_ttf.append(ttf)
        summary_rows.append({"seed": seed, "bo_time_to_threshold": ttf,
                              "bo_final_best": float(curve[-1])})

    # Summary CSV
    pd.DataFrame(summary_rows).to_csv(out_dir / "MS_summary_BO_seeds.csv", index=False)

    ttfs_valid = [t for t in all_ttf if not np.isnan(t)]
    med = float(np.nanmedian(all_ttf))
    std = float(np.nanstd(all_ttf))
    print(f"  TTF median={med:.1f}  std={std:.2f}  "
          f"fails={len(all_ttf)-len(ttfs_valid)}/{len(SEEDS)}")

    # Summary text
    with open(out_dir / "MS_dataset_summary.txt", "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"N={N}  d={d}  budget={B}\n")
        f.write(f"Surrogate: Matérn-2.5 ARD | Acq: LogEI | Init: selected-12\n")
        f.write(f"TTF median={med:.1f}  std={std:.2f}\n")
        f.write(f"Success rate: {len(ttfs_valid)}/{len(SEEDS)}\n")

    # Convergence figure
    x   = np.arange(1, B + 1)
    arr = np.array(all_best)
    m, s = arr.mean(axis=0), arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, m, color="#1f77b4", linewidth=2, label=f"Matérn LogEI ({len(SEEDS)} seeds)")
    ax.fill_between(x, m - s, m + s, color="#1f77b4", alpha=0.15)
    ax.axhline(THRESH, color="orange", linestyle="--", linewidth=1.2,
               label=f"Threshold {THRESH} V")
    ax.axvline(N_INIT, color="k", linestyle=":", linewidth=0.9,
               label=f"Init (n={N_INIT})")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best-so-far E₀ (V vs. SHE)")
    ax.set_title(f"RDKit Matérn convergence — {name}  (d={d})")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "MS_best_so_far_mean_std.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"dataset": name, "d": d, "ttf_median": med, "ttf_std": std,
            "success": len(ttfs_valid)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dir(OUT_ROOT)

    # Collect all datasets to run
    csv_files = sorted(RF_DIR.glob(RDKIT_GLOB))
    if INCLUDE_PCA:
        csv_files += [p for p in PCA_CSVS if p.exists()]

    if not csv_files:
        raise FileNotFoundError(
            f"No RDKit feature CSVs found.\n"
            "Run src/data/rdkit_feature_generation.py first."
        )

    print("Axis 1: RDKit Matérn BO (selected-init, LogEI)")
    print(f"Datasets : {[f.name for f in csv_files]}")
    print(f"Seeds    : {SEEDS}")
    print(f"Budget   : {BUDGET}  |  N_INIT={N_INIT}  |  Thresh={THRESH} V")

    all_summaries = []
    for csv_path in csv_files:
        s = run_dataset(csv_path)
        all_summaries.append(s)

    print("\n" + "=" * 60)
    print("Summary:")
    for s in all_summaries:
        print(f"  {s['dataset']:<55} d={s['d']:>4}  "
              f"TTF={s['ttf_median']:>5.1f} ±{s['ttf_std']:.1f}  "
              f"success={s['success']}/{len(SEEDS)}")


if __name__ == "__main__":
    main()
