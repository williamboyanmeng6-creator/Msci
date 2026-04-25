"""
bo_morgan_matern.py

Axis 1 experiment — Morgan fingerprints with Matérn-2.5 ARD kernel.

This is intentionally the "wrong" kernel-representation pairing, included
to demonstrate the mismatch problem. The Matérn kernel uses squared Euclidean
distances: for binary fingerprint bits this treats "both zero" and "both one"
as equally similar, which loses the structural meaning of the Tanimoto
similarity. The Tanimoto counts shared active bits relative to the union, so
it's much more faithful to what fingerprint similarity means chemically.

In practice this shows up as worse and more variable TTF compared to the
TanimotoGP variant (see bo_tanimoto_morgan.py). k25 Morgan bits + Matérn
achieves 0/10 seeds finding the target — the representation is just too sparse
for Matérn to build a useful distance structure.

Settings match all other Axis 1 experiments exactly:
    - Surrogate     : SingleTaskGP, Matérn-2.5 ARD (BoTorch default)
    - Acquisition   : LogExpectedImprovement
    - Initialisation: selected-init (same 12 molecules, all seeds)
    - Budget        : 100 evaluations
    - Seeds         : 100-109

Inputs:
    data/RF_importance_threshold/Ritsuki_Morgan_RF_top{25,50,100}_withE0.csv
    data/INIT_POOLS/INIT_SELECTED_worst_median_best_by_class.csv

Outputs:
    data/BO_multiseed_MorganRF/<dataset_name>/offline_BO_trace_seed*.csv
    data/BO_multiseed_MorganRF/<dataset_name>/MS_summary_BO_seeds.csv

Usage:
    python src/models/bo_morgan_matern.py
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2] / "data"
RF_DIR    = BASE_DIR / "RF_importance_threshold"
INIT_FILE = BASE_DIR / "INIT_POOLS" / "INIT_SELECTED_worst_median_best_by_class.csv"
OUT_ROOT  = BASE_DIR / "BO_multiseed_MorganRF"

TARGET_COL = "E0_vs_SHE_V"
SID_COL    = "structure_id"
SENTINEL   = -5.0

SEEDS      = list(range(100, 110))
N_INIT     = 12
BUDGET     = 100
THRESH     = 1.30
GP_MAXITER = 50
DEVICE     = "cpu"
DTYPE      = torch.double

EXCLUDE = {TARGET_COL, SID_COL, "Backbone", "Class", "Functional Group",
           "oxidised_smiles", "reduced_smiles", "random_initial_set"}

DATASET_GLOB = "Ritsuki_Morgan_RF_top*_withE0.csv"


# ---------------------------------------------------------------------------
# Utilities (same pattern as bo_rdkit_matern.py)
# ---------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def best_so_far(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)


def time_to_threshold(curve: np.ndarray, thresh: float):
    idx = np.where(curve >= thresh)[0]
    return int(idx[0] + 1) if len(idx) else np.nan


def fit_matern_gp(X_train: torch.Tensor, y_train: torch.Tensor) -> SingleTaskGP:
    """Standard Matérn-2.5 ARD GP (BoTorch SingleTaskGP default)."""
    y_std = standardize(y_train)
    model = SingleTaskGP(X_train, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


def load_selected_init(df: pd.DataFrame) -> list:
    if not INIT_FILE.exists():
        raise FileNotFoundError(
            f"Selected-init pool not found: {INIT_FILE}\n"
            "Run src/data/rdkit_feature_generation.py first."
        )
    init_df   = pd.read_csv(INIT_FILE)
    sid_to_idx = {sid: i for i, sid in enumerate(df[SID_COL])}
    return [sid_to_idx[sid] for sid in init_df[SID_COL] if sid in sid_to_idx]


def run_bo_loop(X_t: torch.Tensor, y_t: torch.Tensor, y: np.ndarray,
                init_idx: list, n_iter: int):
    """
    Matérn GP + LogEI loop. Note that we are deliberately NOT z-scoring X here
    to test whether Matérn can work with raw binary {0,1} inputs. (It can't,
    really — this is the point of the comparison.)
    If you want to z-score binary bits, the mean of each bit is its prevalence
    and the std is close to 0.5. That doesn't meaningfully change the Euclidean
    distance structure because the distances are dominated by rare bits.
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

    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X_df  = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_raw = X_df.values.astype(float)

    X_t  = torch.tensor(X_raw, device=DEVICE, dtype=DTYPE)
    y_t  = torch.tensor(y,     device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    sel_init = load_selected_init(df)

    d = X_t.shape[1]
    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"  N={N}  d={d}  (Morgan bits, raw binary, Matérn kernel)")

    all_best = []
    all_ttf  = []
    summary_rows = []

    for seed in SEEDS:
        print(f"  [seed {seed}] Matérn LogEI (selected-init)...", end=" ", flush=True)
        curve, y_obs = run_bo_loop(X_t, y_t, y, sel_init, n_iter)
        ttf = time_to_threshold(curve, THRESH)
        print(f"TTF={ttf}  final={curve[-1]:.4f}")

        pd.DataFrame({
            "y_observed":  y_obs,
            "best_so_far": curve,
        }).to_csv(out_dir / f"offline_BO_trace_seed{seed}.csv", index=False)

        all_best.append(curve)
        all_ttf.append(ttf)
        summary_rows.append({"seed": seed, "bo_time_to_threshold": ttf,
                              "bo_final_best": float(curve[-1])})

    pd.DataFrame(summary_rows).to_csv(out_dir / "MS_summary_BO_seeds.csv", index=False)

    ttfs_valid = [t for t in all_ttf if not np.isnan(t)]
    med = float(np.nanmedian(all_ttf))
    std = float(np.nanstd(all_ttf))
    print(f"  TTF median={med:.1f}  std={std:.2f}  "
          f"fails={len(all_ttf)-len(ttfs_valid)}/{len(SEEDS)}")

    with open(out_dir / "MS_dataset_summary.txt", "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"N={N}  d={d}  budget={B}\n")
        f.write(f"Surrogate: Matérn-2.5 ARD (kernel-mismatch baseline)\n")
        f.write(f"Acq: LogEI | Init: selected-12\n")
        f.write(f"TTF median={med:.1f}  std={std:.2f}\n")
        f.write(f"Success rate: {len(ttfs_valid)}/{len(SEEDS)}\n")

    # Convergence figure
    x   = np.arange(1, B + 1)
    arr = np.array(all_best)
    m, s = arr.mean(axis=0), arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, m, color="#8C6BB1", linewidth=2,
            label=f"Morgan+Matérn (mismatch, {len(SEEDS)} seeds)")
    ax.fill_between(x, m - s, m + s, color="#8C6BB1", alpha=0.15)
    ax.axhline(THRESH, color="orange", linestyle="--", linewidth=1.2,
               label=f"Threshold {THRESH} V")
    ax.axvline(N_INIT, color="k", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best-so-far E₀ (V vs. SHE)")
    ax.set_title(f"Morgan + Matérn (kernel-mismatch) — {name}  (d={d})")
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

    csv_files = sorted(RF_DIR.glob(DATASET_GLOB))
    if not csv_files:
        raise FileNotFoundError(
            f"No Morgan RF CSVs found in {RF_DIR}.\n"
            "Run src/data/morgan_rf_dataset_generation.py first."
        )

    print("Axis 1 (kernel-mismatch): Morgan fingerprints + Matérn GP")
    print(f"Datasets: {[f.name for f in csv_files]}")

    for csv_path in csv_files:
        run_dataset(csv_path)

    print("\nDone. Outputs in:", OUT_ROOT)


if __name__ == "__main__":
    main()
