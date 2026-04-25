"""
bo_ninit_variation.py

Sweeps the number of initialisation points (n_init) to understand how
sensitive the BO performance is to this hyperparameter.

Motivation: The initialisation size is a practical design choice --
more points give a better initial model but use up a larger fraction
of the fixed evaluation budget. This experiment quantifies that trade-off.

Settings (fixed across all n_init values for fair comparison):
    - Surrogate : SingleTaskGP, Matérn-2.5 ARD
    - Acquisition: Expected Improvement
    - Init pool : hard-init from bottom 90th percentile
    - Budget    : 115 evaluations total
    - Seeds     : 100-109 (10 per n_init value)
    - Threshold : 1.30 V vs. SHE
    - Features  : RF 0.7 dataset (best-performing from Axis 1 experiments)

This script is designed to run on the Imperial cx3 HPC via PBS array jobs.
Each array task handles one value of n_init. For local testing, use --run_all.

Usage:
    # Single n_init (e.g. from PBS array task $n_init=12):
    python src/hpc/bo_ninit_variation.py --n_init 12 --data_dir /path/to/data --out_dir /path/to/outputs

    # All n_init values sequentially (local testing):
    python src/hpc/bo_ninit_variation.py --run_all --data_dir /path/to/data --out_dir /path/to/outputs
"""

import argparse
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.impute import SimpleImputer
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

# ---------------------------------------------------------------------------
# Fixed config (do not change between runs -- only n_init varies)
# ---------------------------------------------------------------------------
N_INIT_VALUES = [3, 6, 9, 12, 15, 20, 25]
TOTAL_BUDGET  = 115
SEEDS         = list(range(100, 110))
N_RAND_BASE   = 50          # random baselines per seed (for comparison)
TOP1_PCT      = 0.01
TOP10_PCT     = 0.10
THRESH        = 1.30        # V -- time-to-threshold target
GP_MAXITER    = 200         # increased from 50 used locally; HPC can afford more
DEVICE        = "cpu"
DTYPE         = torch.double
RF_THR        = "0.7"       # RF feature set to use


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def zscore(X: np.ndarray) -> np.ndarray:
    """Z-score each column; columns with near-zero std are left as-is."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def best_so_far(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)


def time_to_threshold(best_curve: np.ndarray, thresh: float):
    idx = np.where(best_curve >= thresh)[0]
    return int(idx[0] + 1) if len(idx) else np.nan


def make_cutoffs(y: np.ndarray):
    """Compute the key percentile thresholds used throughout the experiment."""
    y_star    = float(np.max(y))
    top1_cut  = float(np.quantile(y, 1 - TOP1_PCT))
    top10_cut = float(np.quantile(y, 1 - TOP10_PCT))
    elite_cut = float(np.quantile(y, 1 - TOP10_PCT))  # hard-init: exclude top 10%
    return y_star, top1_cut, top10_cut, elite_cut


def fit_gp(X_train: torch.Tensor, y_train: torch.Tensor) -> SingleTaskGP:
    y_std = standardize(y_train)
    model = SingleTaskGP(X_train, y_std)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": GP_MAXITER})
    return model


def find_dataset(data_dir: Path, thr: str) -> Path:
    matches = sorted(data_dir.glob(
        f"RF_importance_threshold/Ritsuki_RDKit_RF_importanceMass_{thr}_k*_withE0.csv"
    ))
    if not matches:
        raise FileNotFoundError(
            f"No RF dataset found for threshold {thr} in {data_dir}/RF_importance_threshold/"
        )
    return matches[0]


def build_features(df: pd.DataFrame, target: str, sid: str) -> np.ndarray:
    """Drop non-feature columns, impute any missing values, then z-score."""
    exclude   = {target, sid, "Backbone", "Class", "Functional Group",
                 "oxidised_smiles", "reduced_smiles", "SMILES", "smiles",
                 "random_initial_set"}
    feat_cols = [c for c in df.columns if c not in exclude]
    X_df      = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_df      = X_df.loc[:, ~X_df.isna().all(axis=0)]
    imp       = SimpleImputer(strategy="median")
    X_imp     = imp.fit_transform(X_df.values).astype(float)
    return zscore(X_imp)


# ---------------------------------------------------------------------------
# Single n_init experiment
# ---------------------------------------------------------------------------
def run_one_ninit(n_init: int, csv_path: Path, out_dir: Path, verbose: bool = True) -> dict:
    """
    Run 10-seed BO experiment for a given n_init value.
    Returns a summary dict with median TTF, success count, etc.
    """
    TARGET = "E0_vs_SHE_V"
    SID    = "structure_id"

    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    y_num  = pd.to_numeric(df_raw[TARGET], errors="coerce")
    df     = df_raw.loc[y_num.notna()].copy().reset_index(drop=True)
    y      = pd.to_numeric(df[TARGET], errors="coerce").astype(float).values
    N      = len(df)

    y_star, top1_cut, top10_cut, elite_cut = make_cutoffs(y)
    init_pool = np.where(y < elite_cut)[0]

    if len(init_pool) < n_init:
        raise ValueError(
            f"Init pool has only {len(init_pool)} molecules but n_init={n_init} requested."
        )

    B      = min(TOTAL_BUDGET, N)
    n_iter = B - n_init

    Xz  = build_features(df, TARGET, SID)
    X_t = torch.tensor(Xz, device=DEVICE, dtype=DTYPE)
    y_t = torch.tensor(y,  device=DEVICE, dtype=DTYPE).unsqueeze(-1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"N_INIT={n_init} | N={N}, d={X_t.shape[1]}, budget={B} ({n_iter} BO steps)")
        print(f"y*={y_star:.4f} V | hard-init pool={len(init_pool)}")

    bo_best_mat, bo_reg_mat, bo_h10_mat, bo_h1_mat = [], [], [], []
    bo_ttf, bo_final = [], []
    rand_best_all, rand_reg_all, rand_h10_all, rand_h1_all = [], [], [], []
    rand_ttf_all = []

    for seed in SEEDS:
        t0  = time.time()
        rng = np.random.default_rng(seed)

        init_idx  = rng.choice(init_pool, size=n_init, replace=False).tolist()
        remaining = sorted(set(range(N)) - set(init_idx))

        X_obs      = X_t[init_idx]
        y_obs      = y_t[init_idx]
        chosen     = list(init_idx)
        y_chosen   = y[init_idx].tolist()
        best_curve = best_so_far(np.array(y_chosen)).astype(float)

        # BO loop with Expected Improvement
        for _ in range(n_iter):
            model  = fit_gp(X_obs, y_obs)
            best_f = float(standardize(y_obs).max().item())
            acq    = ExpectedImprovement(model=model, best_f=best_f)

            X_pool = X_t[remaining]
            with torch.no_grad():
                ei_vals = acq(X_pool.unsqueeze(1)).squeeze(-1)

            pick_pos = int(torch.argmax(ei_vals).item())
            nxt      = remaining[pick_pos]

            chosen.append(nxt)
            y_chosen.append(float(y_t[nxt].item()))
            X_obs      = torch.cat([X_obs, X_t[nxt].view(1, -1)], dim=0)
            y_obs      = torch.cat([y_obs, y_t[nxt].view(1, 1)],  dim=0)
            remaining.pop(pick_pos)
            best_curve = np.append(best_curve, max(best_curve[-1], y_chosen[-1]))

        bo_best = best_curve
        bo_reg  = np.maximum(y_star - bo_best, 1e-12)
        bo_h10  = np.cumsum((np.array(y_chosen) >= top10_cut).astype(int))
        bo_h1   = np.cumsum((np.array(y_chosen) >= top1_cut).astype(int))

        bo_best_mat.append(bo_best)
        bo_reg_mat.append(bo_reg)
        bo_h10_mat.append(bo_h10)
        bo_h1_mat.append(bo_h1)
        bo_ttf.append(time_to_threshold(bo_best, THRESH))
        bo_final.append(float(bo_best[-1]))

        # Save per-seed trace
        trace = df.iloc[chosen].copy()
        trace["order"]       = np.arange(1, len(chosen) + 1)
        trace["y_observed"]  = y_chosen
        trace["best_so_far"] = bo_best
        trace.to_csv(out_dir / f"BO_trace_seed{seed}.csv", index=False)

        # Random baselines using the same hard-init pool
        for _ in range(N_RAND_BASE):
            r_init      = rng.choice(init_pool, size=n_init, replace=False).tolist()
            r_remaining = list(set(range(N)) - set(r_init))
            r_follow    = rng.choice(r_remaining, size=(B - n_init), replace=False).tolist()
            ridx        = r_init + r_follow
            y_r         = y[ridx]

            r_best = best_so_far(y_r)
            rand_best_all.append(r_best)
            rand_reg_all.append(np.maximum(y_star - r_best, 1e-12))
            rand_h10_all.append(np.cumsum((y_r >= top10_cut).astype(int)))
            rand_h1_all.append(np.cumsum((y_r >= top1_cut).astype(int)))
            rand_ttf_all.append(time_to_threshold(r_best, THRESH))

        elapsed = time.time() - t0
        if verbose:
            print(f"  [seed {seed}] final={bo_best[-1]:.4f} V  "
                  f"TTF={bo_ttf[-1]}  ({elapsed:.0f}s)")

    # Aggregate results across seeds
    bo_best_mat = np.vstack(bo_best_mat)
    bo_reg_mat  = np.vstack(bo_reg_mat)
    bo_h10_mat  = np.vstack(bo_h10_mat)
    bo_h1_mat   = np.vstack(bo_h1_mat)

    rb_m, rb_s = np.mean(np.vstack(rand_best_all), axis=0), np.std(np.vstack(rand_best_all), axis=0)
    rr_m, rr_s = np.mean(np.vstack(rand_reg_all),  axis=0), np.std(np.vstack(rand_reg_all),  axis=0)
    rh_m, rh_s = np.mean(np.vstack(rand_h10_all),  axis=0), np.std(np.vstack(rand_h10_all),  axis=0)
    r1_m, r1_s = np.mean(np.vstack(rand_h1_all),   axis=0), np.std(np.vstack(rand_h1_all),   axis=0)

    bo_bm, bo_bs = bo_best_mat.mean(axis=0), bo_best_mat.std(axis=0)
    bo_rm, bo_rs = bo_reg_mat.mean(axis=0),  bo_reg_mat.std(axis=0)
    bo_hm, bo_hs = bo_h10_mat.mean(axis=0),  bo_h10_mat.std(axis=0)
    bo_1m, bo_1s = bo_h1_mat.mean(axis=0),   bo_h1_mat.std(axis=0)

    x = np.arange(1, B + 1)

    # Convergence plot
    plt.figure()
    plt.plot(x, bo_bm, label=f"BO (n_init={n_init})")
    plt.fill_between(x, bo_bm - bo_bs, bo_bm + bo_bs, alpha=0.2)
    plt.plot(x, rb_m, label="Random (hard-init)", linestyle="--")
    plt.fill_between(x, rb_m - rb_s, rb_m + rb_s, alpha=0.2)
    plt.axhline(THRESH, color="red", linestyle=":", linewidth=1, label=f"Threshold {THRESH} V")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far E0 (V)")
    plt.title(f"Best-so-far vs Evaluations  [n_init={n_init}]")
    plt.grid(True); plt.legend()
    plt.savefig(out_dir / "best_so_far.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Regret plot (log scale)
    eps = 1e-12
    plt.figure()
    plt.semilogy(x, np.maximum(bo_rm, eps), label=f"BO (n_init={n_init})")
    plt.fill_between(x, np.maximum(bo_rm - bo_rs, eps),
                     np.maximum(bo_rm + bo_rs, eps), alpha=0.2)
    plt.semilogy(x, np.maximum(rr_m, eps), label="Random", linestyle="--")
    plt.fill_between(x, np.maximum(rr_m - rr_s, eps),
                     np.maximum(rr_m + rr_s, eps), alpha=0.2)
    plt.xlabel("Evaluations"); plt.ylabel("Simple regret (V, log scale)")
    plt.title(f"Regret vs Evaluations  [n_init={n_init}]")
    plt.grid(True, which="both"); plt.legend()
    plt.savefig(out_dir / "regret.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Top-10% hits
    plt.figure()
    plt.plot(x, bo_hm, label=f"BO (n_init={n_init})")
    plt.fill_between(x, bo_hm - bo_hs, bo_hm + bo_hs, alpha=0.2)
    plt.plot(x, rh_m, label="Random", linestyle="--")
    plt.fill_between(x, rh_m - rh_s, rh_m + rh_s, alpha=0.2)
    plt.xlabel("Evaluations"); plt.ylabel(f"Cumulative hits (y >= {top10_cut:.3f} V)")
    plt.title(f"Top-10% hits vs Evaluations  [n_init={n_init}]")
    plt.grid(True); plt.legend()
    plt.savefig(out_dir / "hits_top10pct.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Summary CSV and text
    pd.DataFrame({
        "seed":     SEEDS,
        "n_init":   n_init,
        "bo_ttf":   bo_ttf,
        "bo_final": bo_final,
    }).to_csv(out_dir / "summary_seeds.csv", index=False)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Dataset: {csv_path.name}\n")
        f.write(f"N_INIT={n_init} | N={N}, d={X_t.shape[1]}, budget={B}\n")
        f.write(f"BO median TTF = {np.nanmedian(bo_ttf)}\n")
        f.write(f"BO mean TTF = {np.nanmean(bo_ttf):.2f}\n")
        f.write(f"BO final best mean = {np.mean(bo_final):.4f}\n")
        f.write(f"Random median TTF = {np.nanmedian(rand_ttf_all):.1f}\n")
        f.write(f"Seeds reaching threshold: {sum(1 for t in bo_ttf if not np.isnan(t))}/{len(SEEDS)}\n")

    result = {
        "n_init":          n_init,
        "bo_median_ttf":   float(np.nanmedian(bo_ttf)),
        "bo_mean_ttf":     float(np.nanmean(bo_ttf)),
        "bo_final_mean":   float(np.mean(bo_final)),
        "bo_final_std":    float(np.std(bo_final)),
        "rand_median_ttf": float(np.nanmedian(rand_ttf_all)),
        "seeds_hit":       int(sum(1 for t in bo_ttf if not np.isnan(t))),
    }
    if verbose:
        print(f"  -> BO median TTF={result['bo_median_ttf']}  "
              f"Random median TTF={result['rand_median_ttf']}  "
              f"Seeds hit={result['seeds_hit']}/10")
    return result


# ---------------------------------------------------------------------------
# Summary plot across all n_init values
# ---------------------------------------------------------------------------
def make_summary_plot(results: list, out_dir: Path, csv_path: Path = None):
    """
    Key thesis figure: BO median TTF vs n_init, compared against random baseline.
    Also saves a success-rate bar chart and a combined CSV.
    """
    n_inits      = [r["n_init"]          for r in results]
    bo_med_ttf   = [r["bo_median_ttf"]   for r in results]
    rand_med_ttf = [r["rand_median_ttf"] for r in results]

    d_label = ""
    if csv_path is not None:
        m = re.search(r"_k(\d+)_", csv_path.name)
        if m:
            d_label = f", d={m.group(1)}"

    plt.figure(figsize=(7, 4))
    plt.plot(n_inits, bo_med_ttf,   "o-", label="BO (EI, hard-init)", linewidth=2)
    plt.plot(n_inits, rand_med_ttf, "s--", label="Random baseline",   linewidth=2)
    plt.xlabel("Number of initialisation points (n_init)", fontsize=12)
    plt.ylabel("Median TTF (evaluations to reach 1.30 V)", fontsize=12)
    plt.title(f"Effect of initialisation size on BO efficiency\n"
              f"(RF 0.7 features{d_label}, budget=115)", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "SUMMARY_ttf_vs_ninit.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSummary plot saved: {out_dir / 'SUMMARY_ttf_vs_ninit.png'}")

    seeds_hit = [r["seeds_hit"] for r in results]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(n_inits, seeds_hit, color="steelblue", width=1.8)
    plt.axhline(10, color="gray", linestyle="--", linewidth=1, label="Max (10 seeds)")
    for bar, v in zip(bars, seeds_hit):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v), ha="center", va="bottom")
    plt.xlabel("Number of initialisation points (n_init)", fontsize=12)
    plt.ylabel("Seeds reaching 1.30 V (out of 10)", fontsize=12)
    plt.title("Success rate vs initialisation size", fontsize=11)
    plt.legend(); plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "SUMMARY_success_rate_vs_ninit.png", dpi=300, bbox_inches="tight")
    plt.close()

    pd.DataFrame(results).to_csv(out_dir / "SUMMARY_all_ninit.csv", index=False)
    print("Combined summary CSV saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BO n_init variation experiment")
    p.add_argument("--n_init", type=int, default=None,
                   help="n_init value for this run (used by PBS array jobs).")
    p.add_argument("--run_all", action="store_true",
                   help="Run all N_INIT_VALUES sequentially (local testing).")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to the data folder (parent of RF_importance_threshold/).")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Root output directory; sub-folders per n_init are created.")
    p.add_argument("--rf_thr", type=str, default=RF_THR,
                   help=f"RF importance threshold to use (default: {RF_THR}).")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_dataset(data_dir, args.rf_thr)
    print(f"Dataset: {csv_path.name}")

    if args.run_all:
        results = []
        for n_init in N_INIT_VALUES:
            sub_dir = out_dir / f"ninit_{n_init:02d}"
            r = run_one_ninit(n_init, csv_path, sub_dir, verbose=True)
            results.append(r)
        make_summary_plot(results, out_dir, csv_path=csv_path)

    elif args.n_init is not None:
        if args.n_init not in N_INIT_VALUES:
            print(f"Warning: n_init={args.n_init} not in canonical list {N_INIT_VALUES}. Running anyway.")
        sub_dir = out_dir / f"ninit_{args.n_init:02d}"
        run_one_ninit(args.n_init, csv_path, sub_dir, verbose=True)

    else:
        raise ValueError("Provide either --n_init VALUE or --run_all.")


if __name__ == "__main__":
    main()
