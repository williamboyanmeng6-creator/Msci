"""
bo_acquisition_comparison.py

Axis 2 experiment — LogEI vs. UCB acquisition function comparison.

Having established TanimotoGP+Morgan as the best representation in Axis 1,
here I fix the representation to RDKit+Matérn (more stable for this comparison)
and switch the initialisation to *random-init* so that differences in
acquisition behaviour aren't masked by the deterministic selected-init.

Both acquisition functions are run on the same 3 feature sets (RF-thresholded
at τ=0.7/0.8/0.9) so we can check whether the relative ranking is consistent
across dimensionalities.

UCB uses β=2.0 here (the motivated choice from Srinivas et al.'s theoretical
analysis). The UCB β sensitivity sweep is in bo_ucb_beta_sweep.py.

Settings:
    - Surrogate     : Matérn-2.5 ARD (z-scored RDKit features)
    - Acquisition   : LogEI  vs.  UCB (β=2.0)
    - Initialisation: random-init (12 molecules drawn uniformly, per seed)
    - Budget        : 100 evaluations total
    - Seeds         : 100-109

Inputs:
    data/RF_importance_threshold/Ritsuki_RDKit_RF_importanceMass_*_withE0.csv

Outputs (matched to what the figure script expects):
    data/RF_importance_threshold/BO_random_init_10seeds/<dataset>/
        BO_trace_logei_randinit_seed*.csv
        BO_trace_ucb_randinit_seed*.csv

Usage:
    python src/models/bo_acquisition_comparison.py
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
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2] / "data"
RF_DIR    = BASE_DIR / "RF_importance_threshold"
OUT_ROOT  = RF_DIR / "BO_random_init_10seeds"

TARGET_COL = "E0_vs_SHE_V"
SID_COL    = "structure_id"
SENTINEL   = -5.0

SEEDS      = list(range(100, 110))
N_INIT     = 12
BUDGET     = 100
THRESH     = 1.30
GP_MAXITER = 50
UCB_BETA   = 2.0     # theoretical value from Srinivas et al. (2010)
DEVICE     = "cpu"
DTYPE      = torch.double

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
# BO loop (acquisition-agnostic)
# ---------------------------------------------------------------------------
def run_bo_loop(X_t: torch.Tensor, y_t: torch.Tensor, y: np.ndarray,
                init_idx: list, n_iter: int, acq_fn: str):
    """
    Run one BO trial with the specified acquisition function.
    acq_fn: "logei" or "ucb"
    """
    remaining  = [i for i in range(len(y)) if i not in set(init_idx)]
    X_obs      = X_t[init_idx]
    y_obs      = y_t[init_idx]
    y_chosen   = y[init_idx].tolist()
    best_curve = best_so_far(np.array(y_chosen)).astype(float)

    for _ in range(n_iter):
        model = fit_matern_gp(X_obs, y_obs)

        if acq_fn == "logei":
            best_f = float(standardize(y_obs).max().item())
            acq    = LogExpectedImprovement(model=model, best_f=best_f)
        else:
            # UCB operates on the standardised GP posterior, so we evaluate it
            # with the model that was already trained on standardised y
            acq = UpperConfidenceBound(model=model, beta=UCB_BETA)

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

    d = X_t.shape[1]
    print(f"\n{'='*60}")
    print(f"Dataset : {name}  (N={N}, d={d})")

    results = {"logei": [], "ucb": []}

    for seed in SEEDS:
        rng      = np.random.default_rng(seed)
        init_idx = rng.choice(N, size=N_INIT, replace=False).tolist()

        for acq_fn in ("logei", "ucb"):
            print(f"  [seed {seed}] {acq_fn.upper()} random-init...", end=" ", flush=True)
            curve, y_obs = run_bo_loop(X_t, y_t, y, init_idx, n_iter, acq_fn)
            ttf = time_to_threshold(curve, THRESH)
            print(f"TTF={ttf}  final={curve[-1]:.4f}")

            suffix = f"logei_randinit" if acq_fn == "logei" else f"ucb_randinit"
            pd.DataFrame({
                "y_observed":  y_obs,
                "best_so_far": curve,
            }).to_csv(out_dir / f"BO_trace_{suffix}_seed{seed}.csv", index=False)

            results[acq_fn].append({"seed": seed, "ttf": ttf, "final": float(curve[-1])})

    # Print summary
    for acq_fn in ("logei", "ucb"):
        ttfs = [r["ttf"] for r in results[acq_fn]]
        med  = float(np.nanmedian(ttfs))
        std  = float(np.nanstd(ttfs))
        succ = sum(1 for t in ttfs if not np.isnan(t))
        print(f"  {acq_fn.upper():<6} TTF median={med:.1f}  std={std:.2f}  "
              f"success={succ}/{len(SEEDS)}")

    # Comparison figure
    x = np.arange(1, B + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"logei": "#2166AC", "ucb": "#E08214"}
    for acq_fn, col in colors.items():
        curves = []
        for seed in SEEDS:
            suffix = f"logei_randinit" if acq_fn == "logei" else f"ucb_randinit"
            tr = pd.read_csv(out_dir / f"BO_trace_{suffix}_seed{seed}.csv")
            curves.append(tr["best_so_far"].values[:B])
        arr = np.array(curves)
        m, s = arr.mean(axis=0), arr.std(axis=0)
        label = f"LogEI" if acq_fn == "logei" else f"UCB (β={UCB_BETA})"
        ax.plot(x, m, color=col, linewidth=2, label=label)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)

    ax.axhline(THRESH, color="orange", linestyle="--", linewidth=1.2,
               label=f"Threshold {THRESH} V")
    ax.axvline(N_INIT, color="k", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best-so-far E₀ (V vs. SHE)")
    ax.set_title(f"Axis 2: LogEI vs. UCB — {name}")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_best_so_far.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return name


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

    print("Axis 2: LogEI vs. UCB (β=2.0), random-init")
    print(f"Datasets: {[f.name for f in csv_files]}")
    print(f"Seeds   : {SEEDS}  |  Budget: {BUDGET}")

    for csv_path in csv_files:
        run_dataset(csv_path)

    print(f"\nDone. Outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
