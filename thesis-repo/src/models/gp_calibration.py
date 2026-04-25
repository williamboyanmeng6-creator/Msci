"""
gp_calibration.py

GP surrogate calibration study.

Before trusting the BO results, I wanted to check how well the GP surrogates
are actually modelling the landscape. A GP that is badly miscalibrated might
be picking candidates for the wrong reasons (e.g. because it's systematically
overconfident in certain regions).

The calibration protocol:
    - Randomly split the 2625-molecule pool 80/20 (train/test), repeated for
      each of the 10 seeds to quantify variability.
    - Fit the GP on the training set and predict on the test set.
    - Report R², RMSE, MAE, mean predicted std, spread ratio (σ̄/std(y_test)),
      and empirical coverage at credible levels {50%, 68%, 80%, 90%, 95%}.

The spread ratio is perhaps the most informative: a perfectly calibrated GP
would have σ̄/std(y_test) ≈ 1. Both the Matérn and TanimotoGP surrogates turn
out to be overconfident (spread ratio ≈ 0.5), meaning their uncertainty
intervals are too narrow. This is a known issue with GP surrogates fit on
medium-sized datasets — the marginal likelihood tends to pull the noise
variance down, making the model appear more certain than it is.

The result doesn't invalidate the BO findings — overconfident surrogates still
work because they produce consistent rankings of candidates. But it's worth
flagging in the thesis because it suggests the posterior uncertainty shouldn't
be taken literally.

Two models are calibrated:
    - Matérn-2.5 ARD on RDKit top-101 descriptors (z-scored)
    - TanimotoGP on Morgan top-100 bits (raw binary)

Outputs:
    data/BO_GP_calibration/Matern_k101/
        calibration_per_seed.csv
        predictions_seed*.csv
        fig1_parity.png, fig2_calibration_scatter.png, fig3_coverage.png
    data/BO_GP_calibration/TanimotoGP_Morgan100/
        (same structure)

Usage:
    python src/models/gp_calibration.py
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
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2] / "data"
RDKIT_CSV = BASE_DIR / "Ritsuki_full_data_with_rdkit_descriptors.csv"
OUT_ROOT  = BASE_DIR / "BO_GP_calibration"

TARGET_COL  = "E0_vs_SHE_V"
SID_COL     = "structure_id"
SENTINEL    = -5.0
TRAIN_FRAC  = 0.80
TOP_N_RDKIT = 101    # matches the Axis 2/3 k101 feature set
TOP_N_MORGAN= 100    # matches Axis 1 Morgan k100
RF_TREES    = 500
RF_SEED     = 42
GP_MAXITER  = 50
SEEDS       = list(range(100, 110))
DEVICE      = "cpu"
DTYPE       = torch.double

# Credible levels at which we check empirical coverage
COVERAGE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]

EXCLUDE = {TARGET_COL, SID_COL, "Backbone", "Class", "Functional Group",
           "oxidised_smiles", "reduced_smiles", "random_initial_set"}


# ---------------------------------------------------------------------------
# Tanimoto kernel
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
# Feature builders
# ---------------------------------------------------------------------------
def build_rdkit_features(df: pd.DataFrame, y: np.ndarray):
    """Top-101 RDKit descriptors by RF importance, z-scored."""
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.loc[:, ~X_df.isna().all(axis=0)]
    imp  = SimpleImputer(strategy="median")
    X    = imp.fit_transform(X_df.values).astype(float)
    rf   = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=-1)
    rf.fit(X, y)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:TOP_N_RDKIT]
    X_sel   = X[:, top_idx]
    mu, sd  = X_sel.mean(axis=0, keepdims=True), X_sel.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X_sel - mu) / sd


def build_morgan_features(df: pd.DataFrame, y: np.ndarray):
    """Top-100 Morgan bits by RF importance, raw binary (not z-scored)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    rows = []
    for smi in df["oxidised_smiles"]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            rows.append(np.frombuffer(bv.ToBitString().encode(), dtype="uint8") - ord("0"))
        else:
            rows.append(np.zeros(2048, dtype=np.uint8))
    X_ox  = np.vstack(rows).astype(np.float32)

    rows = []
    for smi in df["reduced_smiles"]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            rows.append(np.frombuffer(bv.ToBitString().encode(), dtype="uint8") - ord("0"))
        else:
            rows.append(np.zeros(2048, dtype=np.uint8))
    X_red = np.vstack(rows).astype(np.float32)
    X_all = np.hstack([X_ox, X_red]).astype(float)

    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=-1)
    rf.fit(X_all, y)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:TOP_N_MORGAN]
    return X_all[:, top_idx]


# ---------------------------------------------------------------------------
# GP fitting
# ---------------------------------------------------------------------------
def fit_matern_gp(X_tr: torch.Tensor, y_tr: torch.Tensor) -> SingleTaskGP:
    y_std = standardize(y_tr)
    model = SingleTaskGP(X_tr, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


def fit_tanimoto_gp(X_tr: torch.Tensor, y_tr: torch.Tensor) -> SingleTaskGP:
    y_std = standardize(y_tr)
    covar = gpytorch.kernels.ScaleKernel(TanimotoKernel())
    model = SingleTaskGP(X_tr, y_std, covar_module=covar)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def compute_coverage(y_true: np.ndarray, y_pred: np.ndarray,
                     y_std: np.ndarray, levels: list) -> dict:
    """
    Compute empirical coverage at each credible level.
    For level p, check what fraction of test points fall within the
    (1-p)/2 to 1-(1-p)/2 quantile interval of N(y_pred, y_std^2).
    """
    coverage = {}
    for lev in levels:
        z = stats.norm.ppf(0.5 + lev / 2)
        lo = y_pred - z * y_std
        hi = y_pred + z * y_std
        coverage[lev] = float(np.mean((y_true >= lo) & (y_true <= hi)))
    return coverage


def predict_on_test(model: SingleTaskGP, X_test: torch.Tensor,
                    y_tr: torch.Tensor) -> tuple:
    """
    Get predictive mean and std on test set, un-standardised back to original scale.
    The GP is trained on standardised y, so we need to invert the z-scoring.
    """
    y_mu  = float(y_tr.mean())
    y_sig = float(y_tr.std())
    if y_sig < 1e-12:
        y_sig = 1.0

    model.eval()
    with torch.no_grad():
        post = model.posterior(X_test)
        mu   = post.mean.squeeze(-1).cpu().numpy()
        var  = post.variance.squeeze(-1).cpu().numpy()
    var = np.maximum(var, 0.0)

    # Convert back from standardised scale
    mu_orig  = mu  * y_sig + y_mu
    std_orig = np.sqrt(var) * y_sig
    return mu_orig, std_orig


# ---------------------------------------------------------------------------
# Per-model calibration run
# ---------------------------------------------------------------------------
def run_calibration(model_name: str, X: np.ndarray, y: np.ndarray,
                    use_tanimoto: bool, out_dir: Path):
    ensure_dir(out_dir)
    gp_fn = fit_tanimoto_gp if use_tanimoto else fit_matern_gp

    all_rows = []
    all_preds = []

    for seed in SEEDS:
        rng     = np.random.default_rng(seed)
        N       = len(y)
        n_train = int(N * TRAIN_FRAC)
        idx     = rng.permutation(N)
        tr_idx, te_idx = idx[:n_train], idx[n_train:]

        X_tr = torch.tensor(X[tr_idx], device=DEVICE, dtype=DTYPE)
        y_tr = torch.tensor(y[tr_idx], device=DEVICE, dtype=DTYPE).unsqueeze(-1)
        X_te = torch.tensor(X[te_idx], device=DEVICE, dtype=DTYPE)
        y_te = y[te_idx]

        print(f"  [seed {seed}] fitting {model_name}...", end=" ", flush=True)
        model = gp_fn(X_tr, y_tr)
        mu, sigma = predict_on_test(model, X_te, y_tr.squeeze(-1))

        # Metrics
        residuals = y_te - mu
        r2   = float(1 - np.var(residuals) / np.var(y_te))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae  = float(np.mean(np.abs(residuals)))
        mean_sigma   = float(sigma.mean())
        spread_ratio = mean_sigma / float(np.std(y_te))
        cov = compute_coverage(y_te, mu, sigma, COVERAGE_LEVELS)

        print(f"R²={r2:.3f}  RMSE={rmse:.3f} V  σ̄/std={spread_ratio:.2f}")

        row = {"model": model_name, "seed": seed,
               "n_train": n_train, "n_test": len(te_idx),
               "r2": r2, "rmse": rmse, "mae": mae,
               "mean_sigma": mean_sigma, "spread_ratio": spread_ratio}
        for lev, c in cov.items():
            row[f"coverage_{int(lev*100)}"] = c
        all_rows.append(row)

        # Save per-seed predictions
        pred_df = pd.DataFrame({
            "y_actual":    y_te,
            "y_pred_mean": mu,
            "y_pred_std":  sigma,
            "residual":    residuals,
        })
        pred_df.to_csv(out_dir / f"predictions_seed{seed}.csv", index=False)
        all_preds.append((y_te, mu, sigma))

    cal_df = pd.DataFrame(all_rows)
    cal_df.to_csv(out_dir / "calibration_per_seed.csv", index=False)

    # Print aggregate summary
    print(f"\n  {model_name} calibration summary (10 seeds):")
    for col in ["r2", "rmse", "mae", "spread_ratio"]:
        vals = cal_df[col].values
        print(f"    {col}: {vals.mean():.3f} ± {vals.std():.3f}")
    print("  Coverage (mean):")
    for lev in COVERAGE_LEVELS:
        col  = f"coverage_{int(lev*100)}"
        vals = cal_df[col].values
        print(f"    {lev:.0%}: {vals.mean():.2f} (ideal: {lev:.2f})")

    # Figure 1: parity plot (seed 100)
    y_te_0, mu_0, sig_0 = all_preds[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te_0, mu_0, s=12, alpha=0.6, color="#1f77b4")
    lims = [min(y_te_0.min(), mu_0.min()) - 0.05,
            max(y_te_0.max(), mu_0.max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = ŷ")
    ax.set_xlabel("True E₀ (V vs. SHE)")
    ax.set_ylabel("Predicted E₀ (V vs. SHE)")
    ax.set_title(f"Parity plot — {model_name} (seed 100)")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_parity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: predicted std vs. |residual| (calibration scatter)
    all_y  = np.concatenate([p[0] for p in all_preds])
    all_mu = np.concatenate([p[1] for p in all_preds])
    all_sg = np.concatenate([p[2] for p in all_preds])
    all_re = np.abs(all_y - all_mu)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(all_sg, all_re, s=5, alpha=0.2, color="#1f77b4")
    lim = max(all_sg.max(), all_re.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("Predicted std σ (V)")
    ax.set_ylabel("|Residual| (V)")
    ax.set_title(f"Calibration scatter — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_calibration_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: coverage curve
    mean_cov  = [cal_df[f"coverage_{int(lev*100)}"].mean() for lev in COVERAGE_LEVELS]
    ideal_cov = COVERAGE_LEVELS

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(ideal_cov, mean_cov, "o-", color="#1f77b4", linewidth=2,
            markersize=7, label=f"{model_name}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Coverage calibration — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cal_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dir(OUT_ROOT)

    if not RDKIT_CSV.exists():
        raise FileNotFoundError(f"Base dataset not found: {RDKIT_CSV}")

    df_raw = pd.read_csv(RDKIT_CSV)
    y_raw  = pd.to_numeric(df_raw[TARGET_COL], errors="coerce")
    mask   = y_raw.notna() & (y_raw > SENTINEL)
    df     = df_raw.loc[mask].copy().reset_index(drop=True)
    y      = df[TARGET_COL].astype(float).values
    print(f"Loaded {len(df)} molecules. E0 range [{y.min():.3f}, {y.max():.3f}] V")

    # Matérn on RDKit top-101
    print(f"\n{'='*60}")
    print("Building RDKit top-101 features...")
    X_rdkit = build_rdkit_features(df, y)
    print(f"  Shape: {X_rdkit.shape}")
    run_calibration("Matern_k101", X_rdkit, y, use_tanimoto=False,
                    out_dir=OUT_ROOT / "Matern_k101")

    # TanimotoGP on Morgan top-100
    print(f"\n{'='*60}")
    print("Building Morgan top-100 features...")
    X_morgan = build_morgan_features(df, y)
    print(f"  Shape: {X_morgan.shape}")
    run_calibration("TanimotoGP_Morgan100", X_morgan, y, use_tanimoto=True,
                    out_dir=OUT_ROOT / "TanimotoGP_Morgan100")

    print(f"\nAll calibration outputs saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
