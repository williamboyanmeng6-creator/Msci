"""
bo_initialisation_comparison.py

Axis 3 experiment — hard-init vs. selected-init comparison.

After establishing the best representation (Axis 1) and acquisition function
(Axis 2), I tested two initialisation strategies:

  Hard initialisation:
    12 molecules drawn uniformly at random from the bottom 90th percentile
    of E0. Different random draw per seed → 10 distinct initial sets.
    The key idea is to start far from the optimum so the GP has a strong
    gradient to follow without burning budget on "safe" evaluations.

  Selected (curated) initialisation:
    12 fixed molecules chosen once: worst, median, and best E0 from each
    of the four backbone classes (AQ, BQ, PTZ, PHZ). Identical across all
    seeds → zero inter-seed variance.
    Rationale: structural diversity might help the GP learn the landscape
    better than a random draw from the low-E0 tail.

In practice, hard-init consistently outperforms selected-init in my results.
The likely reason is that the hard-init still explores a much wider chemical
space (random from ~2380 molecules), while selected-init is constrained to
just 12 hand-picked structures.

This script runs both strategies for:
  - RDKit top-k + Matérn (covering BO_comparison_hardinit_vs_selectedinit)
  - TanimotoGP + Morgan top-k (covering BO_multiseed_MorganRF_TanimotoGP_init)

Settings (fixed across both strategy arms):
    - Surrogate : Matérn-2.5 ARD (for RDKit) / TanimotoGP (for Morgan)
    - Acquisition: LogExpectedImprovement
    - Budget    : 100 evaluations
    - Seeds     : 100-109

Inputs:
    data/RF_importance_threshold/Ritsuki_RDKit_RF_importanceMass_*_withE0.csv
    data/RF_importance_threshold/Ritsuki_Morgan_RF_top*_withE0.csv
    data/INIT_POOLS/INIT_SELECTED_worst_median_best_by_class.csv

Outputs:
    data/RF_importance_threshold/BO_comparison_hardinit_vs_selectedinit/<dataset>/
        BO_trace_hard_seed*.csv
        BO_trace_selected_seed*.csv
        comparison_summary.csv
    data/BO_multiseed_MorganRF_TanimotoGP_init/<dataset>/
        BO_trace_hard_seed*.csv
        BO_trace_selected_seed*.csv

Usage:
    python src/models/bo_initialisation_comparison.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2] / "data"
RF_DIR    = BASE_DIR / "RF_importance_threshold"
INIT_FILE = BASE_DIR / "INIT_POOLS" / "INIT_SELECTED_worst_median_best_by_class.csv"

# Outputs for each arm
RDKIT_OUT   = RF_DIR / "BO_comparison_hardinit_vs_selectedinit"
TANIMOTO_OUT = BASE_DIR / "BO_multiseed_MorganRF_TanimotoGP_init"

TARGET_COL = "E0_vs_SHE_V"
SID_COL    = "structure_id"
SENTINEL   = -5.0

SEEDS      = list(range(100, 110))
N_INIT     = 12
BUDGET     = 100
THRESH     = 1.30
ELITE_Q    = 0.90
GP_MAXITER = 50
DEVICE     = "cpu"
DTYPE      = torch.double

EXCLUDE = {TARGET_COL, SID_COL, "Backbone", "Class", "Functional Group",
           "oxidised_smiles", "reduced_smiles", "random_initial_set"}

RDKIT_GLOB   = "Ritsuki_RDKit_RF_importanceMass_*_withE0.csv"
MORGAN_GLOB  = "Ritsuki_Morgan_RF_top*_withE0.csv"


# ---------------------------------------------------------------------------
# Tanimoto kernel (same as in bo_tanimoto_morgan.py)
# ---------------------------------------------------------------------------
class TanimotoKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[-2], device=x1.device, dtype=x1.dtype)
        x1x2  = x1 @ x2.transpose(-1, -2)
        x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)
        x2_sq = (x2 * x2).sum(dim=-1, keepdim=True).transpose(-1, -2)
        return (x1x2 / (x1_sq + x2_sq - x1x2 + self.eps)).clamp(0.0, 1.0)


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


def fit_matern_gp(X_tr: torch.Tensor, y_tr: torch.Tensor) -> SingleTaskGP:
    y_std = standardize(y_tr)
    model = SingleTaskGP(X_tr, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


def fit_tanimoto_gp(X_tr: torch.Tensor, y_tr: torch.Tensor) -> SingleTaskGP:
    """TanimotoGP — X must be raw binary bits (not z-scored)."""
    y_std = standardize(y_tr)
    covar = gpytorch.kernels.ScaleKernel(TanimotoKernel())
    model = SingleTaskGP(X_tr, y_std, covar_module=covar)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


def load_selected_init(df: pd.DataFrame) -> list:
    if not INIT_FILE.exists():
        raise FileNotFoundError(
            f"Selected-init pool not found: {INIT_FILE}\n"
            "Run src/data/rdkit_feature_generation.py first."
        )
    init_df    = pd.read_csv(INIT_FILE)
    sid_to_idx = {sid: i for i, sid in enumerate(df[SID_COL])}
    return [sid_to_idx[sid] for sid in init_df[SID_COL] if sid in sid_to_idx]


def run_bo_loop(X_t: torch.Tensor, y_t: torch.Tensor, y: np.ndarray,
                init_idx: list, n_iter: int, gp_fn):
    remaining  = [i for i in range(len(y)) if i not in set(init_idx)]
    X_obs      = X_t[init_idx]
    y_obs      = y_t[init_idx]
    y_chosen   = y[init_idx].tolist()
    best_curve = best_so_far(np.array(y_chosen)).astype(float)

    for _ in range(n_iter):
        model  = gp_fn(X_obs, y_obs)
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
# Per-dataset, per-surrogate experiment
# ---------------------------------------------------------------------------
def run_dataset(csv_path: Path, out_dir: Path, use_tanimoto: bool):
    name = csv_path.stem
    ensure_dir(out_dir / name)

    df   = pd.read_csv(csv_path)
    y_s  = pd.to_numeric(df[TARGET_COL], errors="coerce")
    mask = y_s.notna() & (y_s > SENTINEL)
    df   = df.loc[mask].copy().reset_index(drop=True)
    y    = df[TARGET_COL].astype(float).values
    N    = len(df)
    B    = min(BUDGET, N)
    n_iter = B - N_INIT

    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_raw = X_df.values.astype(float)

    if use_tanimoto:
        X_proc = X_raw                   # raw bits for TanimotoKernel
        gp_fn  = fit_tanimoto_gp
    else:
        X_proc = zscore_cols(X_raw)      # z-scored for Matérn
        gp_fn  = fit_matern_gp

    X_t = torch.tensor(X_proc, device=DEVICE, dtype=DTYPE)
    y_t = torch.tensor(y,      device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    elite_cut = float(np.quantile(y, ELITE_Q))
    hard_pool = np.where(y < elite_cut)[0]
    sel_init  = load_selected_init(df)

    d       = X_t.shape[1]
    gp_name = "TanimotoGP" if use_tanimoto else "Matérn-2.5 ARD"
    print(f"\n  {name}  (d={d}, {gp_name})")

    results = {"hard": [], "selected": []}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        # Hard-init: random sample from bottom 90%
        init_hard = rng.choice(hard_pool, size=N_INIT, replace=False).tolist()
        # Selected-init: fixed curated 12
        init_sel  = sel_init

        for init_name, init_idx in [("hard", init_hard), ("selected", init_sel)]:
            print(f"    [seed {seed}] {init_name}-init...", end=" ", flush=True)
            curve, y_obs = run_bo_loop(X_t, y_t, y, init_idx, n_iter, gp_fn)
            ttf = time_to_threshold(curve, THRESH)
            print(f"TTF={ttf}")

            pd.DataFrame({
                "y_observed":  y_obs,
                "best_so_far": curve,
            }).to_csv(out_dir / name / f"BO_trace_{init_name}_seed{seed}.csv",
                      index=False)

            results[init_name].append({"seed": seed, "ttf": ttf,
                                        "final": float(curve[-1])})

    # Summary
    summary_rows = []
    for init_name, rows in results.items():
        ttfs = [r["ttf"] for r in rows]
        for row in rows:
            row["condition"] = init_name
        summary_rows.append({
            "condition": init_name,
            "seed":      "all",
            "final_best_mean": float(np.mean([r["final"] for r in rows])),
            "final_best_std":  float(np.std([r["final"] for r in rows])),
            "ttf_median":      float(np.nanmedian(ttfs)),
            "ttf_mean":        float(np.nanmean(ttfs)),
            "ttf_nan_count":   sum(1 for t in ttfs if np.isnan(t)),
        })
        for row in rows:
            summary_rows.append({"condition": row["condition"], "seed": row["seed"],
                                  "final_best_mean": row["final"],
                                  "ttf_median": row["ttf"]})

    pd.DataFrame(summary_rows).to_csv(
        out_dir / name / "comparison_summary.csv", index=False)

    # Comparison figure
    x   = np.arange(1, B + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for init_name, col in [("hard", "#B35806"), ("selected", "#542788")]:
        curves = []
        for seed in SEEDS:
            tr = pd.read_csv(out_dir / name / f"BO_trace_{init_name}_seed{seed}.csv")
            curves.append(tr["best_so_far"].values[:B])
        arr = np.array(curves)
        m, s = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(x, m, linewidth=2, color=col,
                label=f"{init_name}-init (med TTF={float(np.nanmedian([r['ttf'] for r in results[init_name]])):.0f})")
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)

    ax.axhline(THRESH, color="orange", linestyle="--", linewidth=1.2,
               label=f"Threshold {THRESH} V")
    ax.axvline(N_INIT, color="k", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best-so-far E₀ (V vs. SHE)")
    ax.set_title(f"Init strategy comparison ({gp_name}) — {name}")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / name / "comparison_best_so_far.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dir(RDKIT_OUT)
    ensure_dir(TANIMOTO_OUT)

    # Arm 1: RDKit + Matérn (Axis 3 for the importance-mass feature sets)
    rdkit_files = sorted(RF_DIR.glob(RDKIT_GLOB))
    if rdkit_files:
        print("Axis 3 — RDKit + Matérn: hard-init vs. selected-init")
        for csv_path in rdkit_files:
            run_dataset(csv_path, RDKIT_OUT, use_tanimoto=False)
    else:
        print("No RDKit importance-mass CSVs found; skipping RDKit arm.")

    # Arm 2: Morgan top-k + TanimotoGP
    morgan_files = sorted(RF_DIR.glob(MORGAN_GLOB))
    if not morgan_files:
        # Try the main RF directory
        morgan_files = sorted(
            (BASE_DIR / "RF_importance_threshold").glob("Ritsuki_Morgan_RF_top*_withE0.csv")
        )
    if morgan_files:
        print("\nAxis 3 — TanimotoGP + Morgan: hard-init vs. selected-init")
        for csv_path in morgan_files:
            run_dataset(csv_path, TANIMOTO_OUT, use_tanimoto=True)
    else:
        print("No Morgan top-k CSVs found; skipping TanimotoGP arm.")

    print("\nDone.")


if __name__ == "__main__":
    main()
