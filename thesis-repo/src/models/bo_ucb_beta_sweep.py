"""
bo_ucb_beta_sweep.py

UCB acquisition function sensitivity study — sweeps β ∈ {0.1, 0.5, 1.0, 1.5, 2.0}.

β controls the exploration-exploitation trade-off in UCB:
    α_UCB(x) = μ*(x) + β^(1/2) · σ*(x)

Low β → pure exploitation (greedy); high β → heavy exploration.
The Srinivas et al. (2010) theoretical analysis motivates β=2 for Gaussian
priors, but in practice the best β can vary with dataset size and difficulty.

I ran this sweep with hard-init (not random-init) across all three RF-thresholded
feature sets to understand whether the UCB β choice matters in practice. The
result (reported in the thesis) is that performance is relatively stable across
β ∈ {0.5, 1.0, 1.5, 2.0}, with β=0.1 being too greedy and occasionally failing
to find the target.

Settings:
    - Surrogate     : Matérn-2.5 ARD
    - Acquisition   : UCB with β swept over BETA_VALUES
    - Initialisation: hard-init (bottom 90th percentile)
    - Budget        : 100 evaluations
    - Seeds         : 100-109

Outputs (matched to what the figure script expects):
    data/RF_importance_threshold/BO_UCB_beta_sweep_multiseed/<dataset>/
        BO_trace_hard_beta{β}_seed*.csv
        summary_all_betas.csv

Usage:
    python src/models/bo_ucb_beta_sweep.py
    # To run a single β value: set BETA_VALUES = [1.0]
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
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2] / "data"
RF_DIR    = BASE_DIR / "RF_importance_threshold"
OUT_ROOT  = RF_DIR / "BO_UCB_beta_sweep_multiseed"

TARGET_COL = "E0_vs_SHE_V"
SID_COL    = "structure_id"
SENTINEL   = -5.0

SEEDS       = list(range(100, 110))
BETA_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0]
N_INIT      = 12
BUDGET      = 100
THRESH      = 1.30
ELITE_Q     = 0.90   # hard-init: sample from below this percentile
GP_MAXITER  = 50
DEVICE      = "cpu"
DTYPE       = torch.double

EXCLUDE = {TARGET_COL, SID_COL, "Backbone", "Class", "Functional Group",
           "oxidised_smiles", "reduced_smiles", "random_initial_set"}

DATASET_GLOB = "Ritsuki_RDKit_RF_importanceMass_*_withE0.csv"


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
    y_std = standardize(y_train)
    model = SingleTaskGP(X_train, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


# ---------------------------------------------------------------------------
# BO loop (UCB, variable β)
# ---------------------------------------------------------------------------
def run_bo_loop(X_t: torch.Tensor, y_t: torch.Tensor, y: np.ndarray,
                init_idx: list, n_iter: int, beta: float):
    remaining  = [i for i in range(len(y)) if i not in set(init_idx)]
    X_obs      = X_t[init_idx]
    y_obs      = y_t[init_idx]
    y_chosen   = y[init_idx].tolist()
    best_curve = best_so_far(np.array(y_chosen)).astype(float)

    for _ in range(n_iter):
        model = fit_matern_gp(X_obs, y_obs)
        acq   = UpperConfidenceBound(model=model, beta=beta)

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
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]
    imp  = SimpleImputer(strategy="median")
    X    = imp.fit_transform(X_df.values).astype(float)
    X    = zscore_cols(X)

    X_t  = torch.tensor(X, device=DEVICE, dtype=DTYPE)
    y_t  = torch.tensor(y, device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    elite_cut = float(np.quantile(y, ELITE_Q))
    hard_pool = np.where(y < elite_cut)[0]

    d = X_t.shape[1]
    print(f"\n{'='*60}")
    print(f"Dataset : {name}  (N={N}, d={d})")
    print(f"Hard-init pool: {len(hard_pool)} molecules (below {ELITE_Q:.0%})")

    # β × seed grid
    all_results = {beta: [] for beta in BETA_VALUES}

    for beta in BETA_VALUES:
        for seed in SEEDS:
            rng      = np.random.default_rng(seed)
            init_idx = rng.choice(hard_pool, size=N_INIT, replace=False).tolist()

            print(f"  [β={beta}, seed {seed}]...", end=" ", flush=True)
            curve, y_obs = run_bo_loop(X_t, y_t, y, init_idx, n_iter, beta)
            ttf = time_to_threshold(curve, THRESH)
            print(f"TTF={ttf}")

            pd.DataFrame({
                "y_observed":  y_obs,
                "best_so_far": curve,
            }).to_csv(out_dir / f"BO_trace_hard_beta{beta}_seed{seed}.csv", index=False)

            all_results[beta].append(ttf)

    # Summary across β values
    rows = []
    for beta in BETA_VALUES:
        ttfs = all_results[beta]
        med  = float(np.nanmedian(ttfs))
        std  = float(np.nanstd(ttfs))
        succ = sum(1 for t in ttfs if not np.isnan(t))
        rows.append({"beta": beta, "ttf_median": med, "ttf_std": std,
                     "success": succ, "n_seeds": len(SEEDS)})
        print(f"  β={beta}  TTF median={med:.1f}  std={std:.2f}  "
              f"success={succ}/{len(SEEDS)}")

    pd.DataFrame(rows).to_csv(out_dir / "summary_all_betas.csv", index=False)

    # β comparison figure
    betas   = [r["beta"]      for r in rows]
    medians = [r["ttf_median"] for r in rows]
    stds    = [r["ttf_std"]    for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(betas, medians, yerr=stds, fmt="o-", color="#1f77b4",
                linewidth=2, capsize=5, markersize=6)
    ax.axhline(57, color="grey", linestyle="--", linewidth=1,
               alpha=0.8, label="Random baseline (median ≈ 57)")
    ax.set_xlabel("UCB β")
    ax.set_ylabel("Median TTF (evaluations to first hit)")
    ax.set_title(f"UCB β sensitivity — {name}")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "beta_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dir(OUT_ROOT)

    csv_files = sorted(RF_DIR.glob(DATASET_GLOB))
    if not csv_files:
        raise FileNotFoundError(
            f"No RDKit importance-mass CSVs found in {RF_DIR}.\n"
            "Run src/data/rdkit_feature_generation.py first."
        )

    print("UCB β sweep")
    print(f"β values : {BETA_VALUES}")
    print(f"Datasets : {[f.name for f in csv_files]}")
    print(f"Seeds    : {SEEDS}  |  Budget: {BUDGET}")

    for csv_path in csv_files:
        run_dataset(csv_path)

    # Combined summary across all datasets
    all_rows = []
    for csv_path in csv_files:
        name = csv_path.stem
        summary_path = OUT_ROOT / name / "summary_all_betas.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["dataset"] = name
            all_rows.append(df)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            OUT_ROOT / "summary_all_datasets.csv", index=False)

    print(f"\nDone. Outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
