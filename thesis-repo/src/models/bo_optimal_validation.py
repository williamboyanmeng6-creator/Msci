"""
bo_optimal_validation.py

Ablation study validating the claimed optimal BO configuration against
single-factor alternatives. Each condition changes exactly one thing, so
pairwise comparisons isolate individual contributions.

The four conditions are:
    optimal   -- TanimotoGP + Morgan top-100 bits + LogEI + hard-init
    repr_only -- RDKit k101 + Matérn-2.5 ARD + LogEI + hard-init
                 (swaps both kernel and representation simultaneously;
                 Morgan bits really need Tanimoto, RDKit descriptors
                 really need z-scoring + Matérn, so these go together)
    init_only -- TanimotoGP + Morgan top-100 + LogEI + random-init
                 (only the initialisation strategy changes)
    random    -- pure random baseline, no surrogate

All other settings (budget, threshold, seeds) are identical to the main
BO experiments for direct comparability.

Outputs: Dataset/bo_optimal_validation/ (one sub-folder per condition)

Usage:
    python src/models/bo_optimal_validation.py
    # Quick test: set SEEDS = [100] below
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[2] / "data"
RDKIT_CSV  = BASE_DIR / "Ritsuki_full_data_with_rdkit_descriptors.csv"
OUT_ROOT   = BASE_DIR / "bo_optimal_validation"

TARGET_COL   = "E0_vs_SHE_V"
SID_COL      = "structure_id"
SENTINEL     = -5.0

SEEDS        = list(range(100, 110))
N_INIT       = 12
TOTAL_BUDGET = 115
ELITE_Q      = 0.90
THRESH       = 1.30   # V -- defines a "hit"
TOP_N_RDKIT  = 100
TOP_N_MORGAN = 100
GP_MAXITER   = 50
DEVICE       = "cpu"
DTYPE        = torch.double

EXCLUDE = {
    TARGET_COL, SID_COL, "Backbone", "Class", "Functional Group",
    "oxidised_smiles", "reduced_smiles", "random_initial_set",
}

COLORS = {
    "optimal":   "#2ca02c",
    "repr_only": "#1f77b4",
    "init_only": "#ff7f0e",
    "random":    "#7f7f7f",
}
LABELS = {
    "optimal":   "Optimal (TanimotoGP + Morgan100 + hard-init)",
    "repr_only": "RDKit k101 + Matérn + hard-init",
    "init_only": "TanimotoGP + Morgan100 + random-init",
    "random":    "Random baseline",
}


# ---------------------------------------------------------------------------
# Tanimoto kernel (same definition as in bo_tanimoto_morgan.py)
# ---------------------------------------------------------------------------
class TanimotoKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, eps=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[-2], device=x1.device, dtype=x1.dtype)
        x1x2  = x1 @ x2.transpose(-1, -2)
        x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)
        x2_sq = (x2 * x2).sum(dim=-1, keepdim=True).transpose(-1, -2)
        return x1x2 / (x1_sq + x2_sq - x1x2 + self.eps)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def best_so_far(y):
    return np.maximum.accumulate(y)


def time_to_threshold(curve, thresh):
    idx = np.where(curve >= thresh)[0]
    return int(idx[0] + 1) if len(idx) else np.nan


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def build_rdkit_features(df, y):
    """
    Select the top 100 RDKit physicochemical descriptors by RF importance,
    then z-score them. Z-scoring is needed here because Matérn uses
    Euclidean distances in feature space -- scales matter.
    """
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]
    imp  = SimpleImputer(strategy="median")
    X    = imp.fit_transform(X_df.values).astype(float)
    rf   = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:TOP_N_RDKIT]
    X_sel   = X[:, top_idx]
    mu, sd  = X_sel.mean(axis=0, keepdims=True), X_sel.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X_sel - mu) / sd


def _smiles_to_fp(smiles_series, radius=2, n_bits=2048):
    """Compute ECFP4 fingerprints from a SMILES column."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    rows = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            rows.append(np.frombuffer(bv.ToBitString().encode(), dtype="uint8") - ord("0"))
        else:
            rows.append(np.zeros(n_bits, dtype=np.uint8))
    return np.vstack(rows).astype(np.float32)


def build_morgan_features(df, y):
    """
    Select the top 100 Morgan bits by RF importance.
    Computed directly from SMILES rather than loading a pre-computed CSV --
    this avoids a many-to-many explosion caused by non-unique structure_ids
    in some of the pre-computed CSVs.
    Note: NOT z-scored because TanimotoKernel requires raw binary inputs.
    """
    X_ox  = _smiles_to_fp(df["oxidised_smiles"])
    X_red = _smiles_to_fp(df["reduced_smiles"])
    X_all = np.hstack([X_ox, X_red]).astype(float)
    rf    = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_all, y)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:TOP_N_MORGAN]
    return X_all[:, top_idx]


# ---------------------------------------------------------------------------
# GP builders
# ---------------------------------------------------------------------------
def build_tanimoto_gp(X_train, y_train):
    """TanimotoGP: X must be raw binary bits (NOT standardised)."""
    y_std = standardize(y_train)
    covar = ScaleKernel(TanimotoKernel())
    model = SingleTaskGP(X_train, y_std, covar_module=covar)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


def build_matern_gp(X_train, y_train):
    """Standard SingleTaskGP with Matérn-2.5 ARD (BoTorch default)."""
    y_std = standardize(y_train)
    model = SingleTaskGP(X_train, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


# ---------------------------------------------------------------------------
# BO loop
# ---------------------------------------------------------------------------
def run_bo_loop(X_t, y_t, y, init_idx, n_iter, use_tanimoto):
    """
    Generic BO loop -- set use_tanimoto=True for the Tanimoto surrogate,
    False for Matérn.
    X should be z-scored for Matérn, raw binary for Tanimoto.
    """
    remaining  = [i for i in range(len(y)) if i not in set(init_idx)]
    X_obs      = X_t[init_idx]
    y_obs      = y_t[init_idx]
    y_chosen   = y[init_idx].tolist()
    best_curve = best_so_far(np.array(y_chosen)).astype(float)

    gp_fn = build_tanimoto_gp if use_tanimoto else build_matern_gp

    for _ in range(n_iter):
        model  = gp_fn(X_obs, y_obs)
        best_f = float(standardize(y_obs).max().item())
        acq    = LogExpectedImprovement(model=model, best_f=best_f)
        X_pool = X_t[remaining]
        with torch.no_grad():
            vals = acq(X_pool.unsqueeze(1)).squeeze(-1)
        pick_pos = int(torch.argmax(vals).item())
        nxt      = remaining[pick_pos]
        y_val    = float(y_t[nxt].item())
        y_chosen.append(y_val)
        X_obs    = torch.cat([X_obs, X_t[nxt].view(1, -1)], dim=0)
        y_obs    = torch.cat([y_obs, y_t[nxt].view(1, 1)],  dim=0)
        remaining.pop(pick_pos)
        best_curve = np.append(best_curve, max(best_curve[-1], y_val))

    return best_curve, np.array(y_chosen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dir(OUT_ROOT)

    print("Loading dataset...")
    df_raw = pd.read_csv(RDKIT_CSV)
    y_raw  = pd.to_numeric(df_raw[TARGET_COL], errors="coerce")
    mask   = y_raw.notna() & (y_raw > SENTINEL)
    df     = df_raw.loc[mask].copy().reset_index(drop=True)
    y      = df[TARGET_COL].astype(float).values
    N      = len(y)
    B      = min(TOTAL_BUDGET, N)
    n_iter = B - N_INIT

    print(f"  N={N} | y range [{y.min():.3f}, {y.max():.3f}] V")
    print(f"  Hits (>={THRESH} V): {(y >= THRESH).sum()} ({100*(y>=THRESH).mean():.2f}%)")

    # Build feature matrices once; both conditions that need them will reuse
    print("\nBuilding RDKit features (top-100 by RF importance)...")
    X_rdkit = build_rdkit_features(df, y)
    print(f"  RDKit : {X_rdkit.shape}")

    print("Building Morgan features (top-100 bits by RF importance)...")
    X_morgan = build_morgan_features(df, y)
    print(f"  Morgan: {X_morgan.shape}")

    elite_cut = float(np.quantile(y, ELITE_Q))
    hard_pool = np.where(y < elite_cut)[0]

    conditions = {
        "optimal":   {"use_tanimoto": True,  "X": X_morgan},
        "repr_only": {"use_tanimoto": False, "X": X_rdkit},
        "init_only": {"use_tanimoto": True,  "X": X_morgan},
        "random":    {"use_tanimoto": False, "X": X_rdkit},
    }

    all_results = {}

    for cond, cfg in conditions.items():
        print(f"\n{'='*60}\nCondition: {cond.upper()}")
        use_t = cfg["use_tanimoto"]
        X_np  = cfg["X"]
        X_t   = torch.tensor(X_np, device=DEVICE, dtype=DTYPE)
        y_t   = torch.tensor(y,    device=DEVICE, dtype=DTYPE).unsqueeze(-1)

        cond_dir = OUT_ROOT / cond
        ensure_dir(cond_dir)

        best_mat, ttf_list, final_list = [], [], []

        for seed in SEEDS:
            rng = np.random.default_rng(seed)

            # init_only condition: draw from the full pool instead of hard_pool
            if cond == "init_only":
                init_idx = rng.choice(N, size=N_INIT, replace=False).tolist()
            else:
                init_idx = rng.choice(hard_pool, size=N_INIT, replace=False).tolist()

            if cond == "random":
                # Pure random: no surrogate, just pick randomly from remaining
                remaining  = sorted(set(range(N)) - set(init_idx))
                follow_idx = rng.choice(remaining, size=n_iter, replace=False).tolist()
                all_idx    = init_idx + follow_idx
                y_chosen   = y[all_idx]
                b_curve    = best_so_far(y_chosen)
            else:
                b_curve, y_chosen = run_bo_loop(X_t, y_t, y, init_idx, n_iter, use_t)

            ttf = time_to_threshold(b_curve, THRESH)
            best_mat.append(b_curve)
            ttf_list.append(ttf)
            final_list.append(float(b_curve[-1]))

            pd.DataFrame({"y_observed": y_chosen, "best_so_far": b_curve}).to_csv(
                cond_dir / f"trace_seed{seed}.csv", index=False)

            print(f"  [seed {seed}]  TTF={ttf}  final={b_curve[-1]:.4f} V")

        best_mat = np.vstack(best_mat)
        all_results[cond] = {
            "best_mat":   best_mat,
            "ttf_list":   ttf_list,
            "final_list": final_list,
            "ttf_median": float(np.nanmedian(ttf_list)),
            "ttf_std":    float(np.nanstd(ttf_list)),
            "success":    int(sum(1 for t in ttf_list if not np.isnan(t))),
        }

        pd.DataFrame({"seed": SEEDS, "ttf": ttf_list, "final": final_list}).to_csv(
            cond_dir / "per_seed_results.csv", index=False)
        print(f"  Median TTF={all_results[cond]['ttf_median']:.0f}  "
              f"success={all_results[cond]['success']}/{len(SEEDS)}")

    # Plots
    print("\nGenerating figures...")
    x = np.arange(1, B + 1)

    # Fig 1: convergence curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in ["optimal", "repr_only", "init_only", "random"]:
        m  = all_results[cond]["best_mat"].mean(axis=0)
        s  = all_results[cond]["best_mat"].std(axis=0)
        lw = 2.5 if cond == "optimal" else 1.8
        ax.plot(x, m, color=COLORS[cond], linewidth=lw, label=LABELS[cond])
        ax.fill_between(x, m - s, m + s, color=COLORS[cond], alpha=0.15)
    ax.axvline(N_INIT, color="k", linestyle="--", linewidth=0.9,
               alpha=0.5, label=f"Init ends (n={N_INIT})")
    ax.axhline(THRESH, color="red", linestyle=":", linewidth=1.0,
               label=f"Threshold ({THRESH} V)")
    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel("Best-so-far $E_0$ (V vs. SHE)", fontsize=12)
    ax.set_title("Optimal configuration vs. single-factor ablations\n"
                 f"(LogEI, {len(SEEDS)} seeds, mean ± 1 s.d.)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "fig1_best_so_far.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: TTF bar chart with per-seed scatter
    conds   = ["optimal", "repr_only", "init_only", "random"]
    medians = [all_results[c]["ttf_median"] for c in conds]
    stds    = [all_results[c]["ttf_std"]    for c in conds]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(conds))
    bars  = ax.bar(x_pos, medians, color=[COLORS[c] for c in conds], width=0.5, alpha=0.8)
    ax.errorbar(x_pos, medians, yerr=stds, fmt="none",
                color="black", capsize=5, linewidth=1.5)
    for i, cond in enumerate(conds):
        ttfs   = [t for t in all_results[cond]["ttf_list"] if not np.isnan(t)]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(ttfs))
        ax.scatter(np.full(len(ttfs), i) + jitter, ttfs,
                   color=COLORS[cond], edgecolors="white", s=40, zorder=5)
    for bar, med, cond in zip(bars, medians, conds):
        sr = all_results[cond]["success"]
        ax.text(bar.get_x() + bar.get_width() / 2, med + 1.5,
                f"{med:.0f}\n({sr}/{len(SEEDS)})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        ["Optimal\n(Tanimoto+hard)", "RDKit k101\n+hard-init",
         "Tanimoto\n+random-init", "Random\nbaseline"], fontsize=9)
    ax.set_ylabel("Median TTF (evaluations to first hit)", fontsize=11)
    ax.set_title("Time-to-find: optimal vs. single-factor ablations\n"
                 "Error bars ±1 s.d.; dots = per-seed results", fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "fig2_ttf_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Summary CSV
    rows = []
    for cond in conds:
        r = all_results[cond]
        rows.append({
            "condition":    cond,
            "surrogate":    "TanimotoGP" if cond in ("optimal", "init_only") else
                            ("Matern" if cond == "repr_only" else "none"),
            "features":     "Morgan100" if cond in ("optimal", "init_only") else
                            ("RDKit_k101" if cond == "repr_only" else "none"),
            "init":         "random" if cond == "init_only" else "hard_bottom90",
            "ttf_median":   r["ttf_median"],
            "ttf_std":      r["ttf_std"],
            "success_rate": f"{r['success']}/{len(SEEDS)}",
            "final_mean":   float(np.mean(r["final_list"])),
        })
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)

    rand_ttf = all_results["random"]["ttf_median"]
    opt_ttf  = all_results["optimal"]["ttf_median"]
    repr_ttf = all_results["repr_only"]["ttf_median"]
    init_ttf = all_results["init_only"]["ttf_median"]

    print(f"\n{'='*60}\nSUMMARY")
    for r in rows:
        speedup = rand_ttf / r["ttf_median"] if r["ttf_median"] > 0 else float("inf")
        print(f"  {r['condition']:12s}  TTF={r['ttf_median']:5.0f}"
              f"  std={r['ttf_std']:4.1f}"
              f"  success={r['success_rate']}"
              f"  speedup={speedup:.2f}x vs random")
    print(f"\n  Representation effect: -{repr_ttf - opt_ttf:.0f} iters (Matern -> Tanimoto)")
    print(f"  Initialisation effect: -{init_ttf - opt_ttf:.0f} iters (random -> hard-init)")
    print(f"  Combined speedup over random: {rand_ttf / opt_ttf:.2f}x")
    print(f"\nOutputs saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
