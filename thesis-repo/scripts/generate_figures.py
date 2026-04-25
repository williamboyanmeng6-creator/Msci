"""
generate_figures.py

Generates all thesis figures from the BO experiment outputs.

Run this after all BO experiments have completed and the results CSVs are
in place. Figures are saved as PDF (300 dpi) to ./figures_out/.

Figures produced:
    fig5  -- Axis 1: TTF by representation type (RDKit | Morgan Matérn | TanimotoGP | PCA)
    fig6  -- Axis 2: initialisation strategy comparison (hard-init vs. selected-init)
    fig7  -- Axis 3: acquisition function comparison (LogEI vs. UCB β=1.0)
    fig9  -- Convergence curves (mean ± 1σ best-so-far + fraction of seeds having hit target)
    SI1   -- Supplementary per-seed TTF heatmap across all conditions

Usage:
    python scripts/generate_figures.py
    # Figures go to ./figures_out/
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Global matplotlib style
matplotlib.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 14,
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 14,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
    "legend.fontsize"  : 13,
    "legend.framealpha": 0.93,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

PANEL_LABEL_SIZE = 16  # "(a)", "(b)", etc.

# Paths -- update DATASET_DIR if running from a different location
DATA_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DATA_ROOT, "..", "data")
OUT_DIR     = os.path.join(DATA_ROOT, "..", "figures_out")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 1.30   # V vs. SHE -- the "hit" criterion
BUDGET    = 100
RNG       = np.random.default_rng(42)

# Colour palette used across figures
C = dict(
    tanimoto   = "#2166AC",
    matern_rdk = "#D6604D",
    pca        = "#4DAC26",
    matern_mor = "#8C6BB1",
    hard       = "#B35806",
    selected   = "#542788",
    random_bl  = "#636363",
    ucb        = "#E08214",
)

# Colours and linestyles specifically for Fig 9 convergence curves
F9_COLS = {
    "TanimotoGP k100"     : "#1A5276",
    "TanimotoGP k50"      : "#5DADE2",
    "RDKit k100"          : "#C0392B",
    "RDKit k25"           : "#E59866",
    "Morgan k50 (Matérn)" : "#7D3C98",
    "PCA 2D"              : "#1E8449",
}
F9_LS = {
    "TanimotoGP k100"     : "-",
    "TanimotoGP k50"      : "--",
    "RDKit k100"          : "-",
    "RDKit k25"           : "--",
    "Morgan k50 (Matérn)" : "-.",
    "PCA 2D"              : ":",
}

FAIL_MARKER_COL = "#C0392B"  # used in Fig 7 for failed seeds


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ms_summary(path):
    """Load TTF values from a multi-seed summary CSV."""
    df  = pd.read_csv(path)
    col = next(c for c in df.columns if "threshold" in c.lower() or "ttf" in c.lower())
    return df[col].values.astype(float)


def _ttf_from_trace(path):
    """Compute TTF from a single-seed best_so_far trace CSV."""
    bsf  = pd.read_csv(path)["best_so_far"].values
    hits = np.where(bsf >= THRESHOLD)[0]
    return int(hits[0]) + 1 if len(hits) else np.nan


def _ttf_glob(pattern, budget=None):
    """
    Load TTF from all trace files matching a glob pattern.
    If budget is set, any TTF > budget counts as a failure (nan).
    Returns a length-10 array; unfound files give nan.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return np.full(10, np.nan)
    ttfs = np.array([_ttf_from_trace(f) for f in files], dtype=float)
    if budget is not None:
        ttfs[ttfs > budget] = np.nan
    return ttfs


def _conv_stats_mean(pattern, max_iter):
    """
    Load all seed traces matching a pattern, pad to max_iter if needed,
    and return (t, mean, std).
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None, None
    mats = []
    for p in files:
        a = pd.read_csv(p)["best_so_far"].values
        if len(a) < max_iter:
            a = np.pad(a, (0, max_iter - len(a)), constant_values=a[-1])
        mats.append(a[:max_iter])
    m = np.array(mats)
    t = np.arange(1, max_iter + 1)
    return t, m.mean(axis=0), m.std(axis=0)


def _boxplot(ax, data, pos, width, col, hatch="", median_col="white"):
    """Draw a boxplot with overlaid scatter jitter at position pos."""
    valid = np.asarray(data, float)
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return False
    ax.boxplot(
        valid, positions=[pos], widths=width, patch_artist=True,
        boxprops    =dict(facecolor=col, alpha=0.75, hatch=hatch),
        medianprops =dict(color=median_col, linewidth=2.2),
        whiskerprops=dict(color=col, linewidth=1.4),
        capprops    =dict(color=col, linewidth=1.4),
        flierprops  =dict(marker="o", ms=4, markerfacecolor=col, alpha=0.5),
        zorder=3,
    )
    jit = RNG.uniform(-width * 0.22, width * 0.22, size=len(valid))
    ax.scatter(pos + jit, valid, color=col, s=14, alpha=0.50, zorder=4)
    return True


def _panel_label(ax, letter, x=0.015, y=0.97):
    """Add a bold panel letter (a), (b), etc. in the top-left corner."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


def _save(fig, name):
    out = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Axis 1: TTF by Representation
# Groups: RDKit (Matérn) | Morgan (Matérn) | TanimotoGP | PCA (Matérn)
# ---------------------------------------------------------------------------

def fig5_axis1_representation():
    print("Fig 5: Axis 1 – Representation TTF...")

    def _load(sub, ds):
        p = f"{DATASET_DIR}/{sub}/{ds}/MS_summary_BO_seeds.csv"
        return _ms_summary(p) if os.path.exists(p) else np.full(10, np.nan)

    groups = [
        dict(label="RDKit (Matérn)", short="RDKit\n(Matérn)", col=C["matern_rdk"],
             configs=[
                 ("k25",  _load("BO_multiseed_RDKIT", "Ritsuki_RDKit_RF_top25_withE0")),
                 ("k50",  _load("BO_multiseed_RDKIT", "Ritsuki_RDKit_RF_top50_withE0")),
                 ("k100", _load("BO_multiseed_RDKIT", "Ritsuki_RDKit_RF_top100_withE0")),
             ]),
        dict(label="Morgan (Matérn)", short="Morgan\n(Matérn)", col=C["matern_mor"],
             configs=[
                 ("k25",  np.full(10, np.nan)),
                 ("k50",  _load("BO_multiseed_MorganRF", "Ritsuki_Morgan_RF_top50_withE0")),
                 ("k100", _load("BO_multiseed_MorganRF", "Ritsuki_Morgan_RF_top100_withE0")),
             ]),
        dict(label="TanimotoGP", short="TanimotoGP", col=C["tanimoto"],
             configs=[
                 ("k25",  _load("BO_multiseed_MorganRF_TanimotoGP", "Ritsuki_Morgan_RF_top25_withE0")),
                 ("k50",  _load("BO_multiseed_MorganRF_TanimotoGP", "Ritsuki_Morgan_RF_top50_withE0")),
                 ("k100", _load("BO_multiseed_MorganRF_TanimotoGP", "Ritsuki_Morgan_RF_top100_withE0")),
             ]),
        dict(label="PCA (Matérn)", short="PCA\n(Matérn)", col=C["pca"],
             configs=[
                 ("2D",   _load("BO_multiseed_RDKIT", "Ritsuki_dataset_RDKitcleared_PCA_fixed")),
                 ("90%",  _load("BO_multiseed_RDKIT", "Ritsuki_dataset_RDKitcleared_PCA_90")),
                 ("95%",  _load("BO_multiseed_RDKIT", "Ritsuki_dataset_RDKitcleared_PCA_95")),
             ]),
    ]

    Y_CEIL = 115
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.subplots_adjust(bottom=0.22)

    x_pos = 0.0
    tick_x, tick_l = [], []
    grp_cx, grp_lb, grp_col = [], [], []

    for g in groups:
        g_start = x_pos
        for lbl, ttfs in g["configs"]:
            valid  = ttfs[~np.isnan(ttfs)]
            n_fail = int(np.sum(np.isnan(ttfs)))

            if len(valid) == 0:
                ax.text(x_pos, 8, "FAIL\n(0/10)", ha="center", va="bottom",
                        fontsize=11, color="firebrick", fontweight="bold",
                        linespacing=1.3)
            else:
                _boxplot(ax, valid, x_pos, 0.62, g["col"], median_col="black")
                if n_fail > 0:
                    y_top = np.nanmax(valid)
                    y_ann = Y_CEIL * 0.78 if y_top > Y_CEIL * 0.85 else min(y_top + 3, Y_CEIL * 0.96)
                    ax.text(x_pos, y_ann, f"{n_fail}/10 FAIL",
                            ha="center", va="bottom", fontsize=11,
                            color="firebrick", fontweight="bold",
                            bbox=dict(fc="white", ec="firebrick",
                                      alpha=0.90, boxstyle="round,pad=0.2"))

            tick_x.append(x_pos); tick_l.append(lbl)
            x_pos += 1.25

        grp_cx.append((g_start + x_pos - 1.25) / 2)
        grp_lb.append(g["label"])
        grp_col.append(g["col"])
        x_pos += 0.75

    ax.axhline(57, color=C["random_bl"], lw=1.5, ls="--", alpha=0.85,
               label="Random baseline (cond. median ≈ 57)")
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_l, fontsize=13)
    ax.set_ylabel("Time-to-Find (iterations, lower is better)")
    ax.set_ylim(0, Y_CEIL)
    ax.set_xlim(-0.7, x_pos - 0.75)
    ax.legend(loc="upper right", fontsize=13)

    for cx, lb, col in zip(grp_cx, grp_lb, grp_col):
        ax.annotate("", xy=(cx + 1.5 * 1.25 / 2, -0.14),
                    xytext=(cx - 1.5 * 1.25 / 2, -0.14),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=col, lw=2.5))
        ax.text(cx, -0.20, lb, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=13, fontweight="bold", color=col)

    fig.tight_layout()
    _save(fig, "fig5_axis1_representation")


# ---------------------------------------------------------------------------
# Figure 6 — Initialisation Strategy Comparison
# Both panels share y-axis [14, 90]; speedup arrows removed.
# ---------------------------------------------------------------------------

def fig6_init_comparison():
    print("Fig 6: Initialisation comparison...")

    rdkit_sel = {
        "k25":  _ms_summary(f"{DATASET_DIR}/BO_multiseed_RDKIT/Ritsuki_RDKit_RF_top25_withE0/MS_summary_BO_seeds.csv"),
        "k50":  _ms_summary(f"{DATASET_DIR}/BO_multiseed_RDKIT/Ritsuki_RDKit_RF_top50_withE0/MS_summary_BO_seeds.csv"),
        "k100": _ms_summary(f"{DATASET_DIR}/BO_multiseed_RDKIT/Ritsuki_RDKit_RF_top100_withE0/MS_summary_BO_seeds.csv"),
    }
    # Hard-init values from Table 8 (no trace CSVs for these runs)
    rdkit_hard = {
        "k100": np.array([30, 19, 29, 39, 19, 19, 27, 22, 17, 23], float),
        "k50":  np.array([33, 19, 35, 35, 35, 20, 37, 54, 32, 19], float),
        "k25":  np.array([21, 24, 31, 37, 24, 23, 26, 21, 25, 37], float),
    }
    _ti = f"{DATASET_DIR}/BO_multiseed_MorganRF_TanimotoGP_init"
    tan_sel = {
        "k50":  _ms_summary(f"{DATASET_DIR}/BO_multiseed_MorganRF_TanimotoGP/Ritsuki_Morgan_RF_top50_withE0/MS_summary_BO_seeds.csv"),
        "k100": _ms_summary(f"{DATASET_DIR}/BO_multiseed_MorganRF_TanimotoGP/Ritsuki_Morgan_RF_top100_withE0/MS_summary_BO_seeds.csv"),
    }
    tan_hard = {
        "k50":  _ttf_glob(f"{_ti}/Ritsuki_Morgan_RF_top50_withE0/BO_trace_hard_seed*.csv"),
        "k100": _ttf_glob(f"{_ti}/Ritsuki_Morgan_RF_top100_withE0/BO_trace_hard_seed*.csv"),
    }
    # Fallback to Table 8 values if trace files not present
    if np.all(np.isnan(tan_hard["k100"])):
        tan_hard["k100"] = np.array([23, 21, 52, 24, 15, 20, 37, 22, 14, 15], float)
    if np.all(np.isnan(tan_hard["k50"])):
        tan_hard["k50"]  = np.array([33, 15, 33, 29, 18, 18, 37, 35, 15, 15], float)

    Y_MIN, Y_MAX = 14, 90
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    w = 0.30

    # Panel (a): RDKit Matérn k25/k50/k100
    keys_a  = ["k25", "k50", "k100"]
    xlbls_a = [f"RDKit {k}" for k in keys_a]
    xs_a    = np.arange(len(keys_a))

    for i, k in enumerate(keys_a):
        _boxplot(ax1, rdkit_hard[k], xs_a[i] - w/2 - 0.03, w, C["hard"], median_col="white")
        _boxplot(ax1, rdkit_sel[k],  xs_a[i] + w/2 + 0.03, w, C["selected"], hatch="///", median_col="white")
        med_s = np.nanmedian(rdkit_sel[k])
        sel_valid = rdkit_sel[k][~np.isnan(rdkit_sel[k])]
        if len(sel_valid):
            y_ann = min(float(np.nanmax(sel_valid)) + 1.5, Y_MAX * 0.96)
            ax1.text(xs_a[i] + w/2 + 0.03, y_ann, f"{med_s:.0f}",
                     ha="center", va="bottom", fontsize=12, color=C["selected"])

    ax1.axhline(57, color=C["random_bl"], lw=1.5, ls="--", alpha=0.85,
                label="Random baseline (cond. median ≈ 57)")
    ax1.set_xticks(xs_a)
    ax1.set_xticklabels(xlbls_a, fontsize=13)
    ax1.set_ylabel("Time-to-Find (iterations, lower is better)")
    ax1.set_ylim(Y_MIN, Y_MAX)
    ax1.legend(handles=[
        mpatches.Patch(fc=C["hard"],     alpha=0.75, label="Hard init (random from bottom-90% $E_0$)"),
        mpatches.Patch(fc=C["selected"], alpha=0.75, hatch="///",
                       label="Selected init (fixed curated set, deterministic)"),
        mpatches.Patch(fc=C["random_bl"], alpha=0.6, label="Random baseline (cond. median ≈ 57)"),
    ], fontsize=12, loc="upper right")
    _panel_label(ax1, "a")

    # Panel (b): Tanimoto k50/k100
    keys_b  = ["k50", "k100"]
    xlbls_b = [f"Morgan {k}" for k in keys_b]
    xs_b    = np.arange(len(keys_b))

    for i, k in enumerate(keys_b):
        _boxplot(ax2, tan_hard[k], xs_b[i] - w/2 - 0.03, w, C["hard"], median_col="white")
        _boxplot(ax2, tan_sel[k],  xs_b[i] + w/2 + 0.03, w, C["selected"], hatch="///", median_col="white")
        med_s = np.nanmedian(tan_sel[k])
        sel_valid = tan_sel[k][~np.isnan(tan_sel[k])]
        if len(sel_valid):
            y_ann = min(float(np.nanmax(sel_valid)) + 1.0, Y_MAX * 0.96)
            ax2.text(xs_b[i] + w/2 + 0.03, y_ann, f"{med_s:.0f}",
                     ha="center", va="bottom", fontsize=12, color=C["selected"])

    ax2.axhline(57, color=C["random_bl"], lw=1.5, ls="--", alpha=0.85)
    ax2.set_xticks(xs_b)
    ax2.set_xticklabels(xlbls_b, fontsize=13)
    ax2.set_ylabel("Time-to-Find (iterations, lower is better)")
    ax2.set_ylim(Y_MIN, Y_MAX)
    _panel_label(ax2, "b")

    fig.tight_layout()
    _save(fig, "fig6_init_comparison")


# ---------------------------------------------------------------------------
# Figure 7 — Acquisition Function Comparison (LogEI vs. UCB β=1.0)
# Failed seeds shown as × markers rather than text boxes.
# ---------------------------------------------------------------------------

def fig7_acqfn_comparison():
    print("Fig 7: Acquisition function comparison...")

    _src_logei = f"{DATASET_DIR}/RF_importance_threshold/BO_random_init_10seeds"
    _src_ucb   = f"{DATASET_DIR}/RF_importance_threshold/BO_UCB_beta_sweep_multiseed"

    feature_sets = [
        ("k100 (d = 100)",
         "Ritsuki_RDKit_RF_importanceMass_0.7_k101_withE0",
         "Ritsuki_RDKit_RF_importanceMass_0.7_k101_withE0"),
        ("k50  (d = 50)",
         "Ritsuki_RDKit_RF_importanceMass_0.8_k68_withE0",
         "Ritsuki_RDKit_RF_importanceMass_0.8_k68_withE0"),
        ("k25  (d = 25)",
         "Ritsuki_RDKit_RF_importanceMass_0.9_k34_withE0",
         "Ritsuki_RDKit_RF_importanceMass_0.9_k34_withE0"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=False)
    any_fail_overall = False

    for ai, (lbl, logei_fld, ucb_fld) in enumerate(feature_sets):
        ax = axes[ai]

        logei = _ttf_glob(f"{_src_logei}/{logei_fld}/BO_trace_logei_randinit_seed*.csv", budget=BUDGET)
        ucb   = _ttf_glob(f"{_src_ucb}/{ucb_fld}/BO_trace_hard_beta1.0_seed*.csv", budget=BUDGET)

        Y_MAX = 80
        w     = 0.30
        panel_has_fail = False

        for data, xpos, col, hatch in [
            (logei, 0.0, C["tanimoto"], ""),
            (ucb,   0.6, C["ucb"],      "///"),
        ]:
            valid  = data[~np.isnan(data)]
            n_fail = int(np.sum(np.isnan(data)))
            if len(valid):
                _boxplot(ax, data, xpos, w, col, hatch=hatch, median_col="white")
                med = float(np.nanmedian(valid))
                ax.text(xpos, med - 1.5, f"{med:.0f}",
                        ha="center", va="top", fontsize=12,
                        color="white", fontweight="bold")
                if n_fail > 0:
                    y_top = float(np.nanmax(valid))
                    Y_MAX = max(Y_MAX, y_top + 16)
                    ax.scatter([xpos], [y_top + 5], marker="x",
                               color=FAIL_MARKER_COL, s=130, linewidths=2.5,
                               zorder=6, clip_on=False)
                    panel_has_fail = True
                    any_fail_overall = True
            elif len(data) > 0:
                Y_MAX = max(Y_MAX, 90)
                ax.scatter([xpos], [Y_MAX * 0.88], marker="x",
                           color=FAIL_MARKER_COL, s=130, linewidths=2.5,
                           zorder=6, clip_on=False)
                panel_has_fail = True
                any_fail_overall = True

        ax.axhline(57, color=C["random_bl"], lw=1.5, ls="--", alpha=0.85)
        ax.set_xticks([0.0, 0.6])
        ax.set_xticklabels(["LogEI", "UCB (β = 1.0)"], fontsize=13)
        ax.set_xlabel(lbl, fontsize=13)
        if ai == 0:
            ax.set_ylabel("Time-to-Find (iterations, lower is better)")
        _panel_label(ax, chr(ord("a") + ai))
        ax.set_ylim(0, max(ax.get_ylim()[1], Y_MAX))

    handles = [
        mpatches.Patch(fc=C["tanimoto"], alpha=0.75, label="LogEI (Log Expected Improvement)"),
        mpatches.Patch(fc=C["ucb"], alpha=0.75, hatch="///", label="UCB (β = 1.0)"),
        Line2D([0], [0], color=C["random_bl"], lw=1.5, ls="--",
               label="Random baseline (cond. median ≈ 57)"),
    ]
    if any_fail_overall:
        handles.append(Line2D([0], [0], marker="x", color=FAIL_MARKER_COL,
                              lw=0, markersize=10, markeredgewidth=2.5,
                              label="No hit within 100-iteration budget"))

    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(handles), fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    _save(fig, "fig7_acqfn_comparison")


# ---------------------------------------------------------------------------
# Figure 9 — Convergence Curves
# Panel (a): mean best-so-far ± 1σ
# Panel (b): fraction of seeds having found ≥1 target (step function)
# ---------------------------------------------------------------------------

def fig9_convergence():
    print("Fig 9: Convergence curves...")

    configs = [
        ("TanimotoGP k100",
         f"{DATASET_DIR}/BO_multiseed_MorganRF_TanimotoGP"
         f"/Ritsuki_Morgan_RF_top100_withE0/offline_BO_trace_seed*.csv"),
        ("TanimotoGP k50",
         f"{DATASET_DIR}/BO_multiseed_MorganRF_TanimotoGP"
         f"/Ritsuki_Morgan_RF_top50_withE0/offline_BO_trace_seed*.csv"),
        ("RDKit k100",
         f"{DATASET_DIR}/BO_multiseed_RDKIT"
         f"/Ritsuki_RDKit_RF_top100_withE0/offline_BO_trace_seed*.csv"),
        ("RDKit k25",
         f"{DATASET_DIR}/BO_multiseed_RDKIT"
         f"/Ritsuki_RDKit_RF_top25_withE0/offline_BO_trace_seed*.csv"),
        ("Morgan k50 (Matérn)",
         f"{DATASET_DIR}/BO_multiseed_MorganRF"
         f"/Ritsuki_Morgan_RF_top50_withE0/offline_BO_trace_seed*.csv"),
        ("PCA 2D",
         f"{DATASET_DIR}/BO_multiseed_RDKIT"
         f"/Ritsuki_dataset_RDKitcleared_PCA_fixed/offline_BO_trace_seed*.csv"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    legend_handles = []

    for name, pattern in configs:
        col = F9_COLS[name]
        ls  = F9_LS[name]

        # Panel (a): mean ± 1σ best-so-far
        t, mn, sd = _conv_stats_mean(pattern, BUDGET)
        if t is not None:
            ax1.plot(t, mn, color=col, ls=ls, lw=2.0, label=name)
            ax1.fill_between(t, mn - sd, mn + sd, color=col, alpha=0.12)

        # Panel (b): fraction of seeds that have found ≥1 target by iteration t
        files = sorted(glob.glob(pattern))
        if files:
            n_seeds  = len(files)
            ttf_vals = np.array([_ttf_from_trace(f) for f in files], dtype=float)
            frac     = np.zeros(BUDGET)
            for s_ttf in ttf_vals:
                if not np.isnan(s_ttf):
                    idx = int(s_ttf) - 1
                    if idx < BUDGET:
                        frac[idx] += 1
            frac_cum = np.cumsum(frac) / n_seeds
            ax2.step(np.arange(1, BUDGET + 1), frac_cum, where="post",
                     color=col, ls=ls, lw=2.0)

        legend_handles.append(Line2D([0], [0], color=col, ls=ls, lw=2.0, label=name))

    ax1.axhline(THRESHOLD, color="black", lw=1.5, ls=":",
                label=f"$E^0$ threshold = {THRESHOLD} V")
    ax1.set_xlabel("BO Iteration")
    ax1.set_ylabel("Best $E^0$ Found So Far (V vs. SHE)")
    ax1.set_xlim(0, BUDGET)
    ax1.set_ylim(0.2, 1.43)
    _panel_label(ax1, "a")

    # Theoretical random baseline for panel (b)
    ax2.plot([0, BUDGET], [0.0, 0.354], color="#888888", ls=":", lw=1.8,
             label="Random (theoretical)")
    ax2.set_xlabel("BO Iteration")
    ax2.set_ylabel("Fraction of Seeds Having Found ≥1 Target")
    ax2.set_xlim(0, BUDGET)
    ax2.set_ylim(0, 1.07)
    _panel_label(ax2, "b")

    legend_handles += [
        Line2D([0], [0], color="#888888", ls=":", lw=1.8, label="Random (theoretical)"),
        Line2D([0], [0], color="black",   ls=":", lw=1.5,
               label=f"$E^0$ threshold = {THRESHOLD} V"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=13, framealpha=0.93)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    _save(fig, "fig9_convergence")


# ---------------------------------------------------------------------------
# SI Figure 1 — Per-seed TTF heatmap
# Two stacked panels; amber for TTF 80-99; grey for FAIL.
# ---------------------------------------------------------------------------

def SI_fig1_perseed_ttf_heatmap():
    print("SI Fig 1: Per-seed TTF heatmap...")

    seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

    # All values taken from Table 8 of the thesis
    rows_top = [
        ("RDKit k25 (Sel.)",           [26, 24, 45, 34, 28, 46, 25, 29, 21, 28]),
        ("RDKit k50 (Sel.)",           [21, 44, 42, 24, 31, 25, 25, 39, 47, 39]),
        ("RDKit k100 (Sel.)",          [26, 22, 28, 38, 21, 29, 39, 17, 56, 23]),
        ("PCA 2D (Sel.)",              [19, 20, 31, 46, 17, 41, 86, 16, 86, 20]),
        ("PCA 90% (Sel.)",             [26, 36, 34, 41, 40, 33, 32, 24, 27, 19]),
        ("PCA 95% (Sel.)",             [34, 40, 42, 36, 35, 37, 43, 20, 32, 20]),
        ("Morgan k50, Matérn (Sel.)",  [47, 24, 28, 25, 32, 46, 18, 20, 26, 20]),
        ("Morgan k100, Matérn (Sel.)", [63, 30, 40, 97, 44, 69, 18, 20, 71, 18]),
        ("Tanimoto k50 (Sel.)",        [23, 25, 26, 25, 20, 35, 26, 17, 20, 17]),
        ("Tanimoto k100 (Sel.)",       [20, 26, 23, 20, 30, 25, 23, 20, 26, 17]),
    ]
    rows_bot = [
        ("RDKit k100, LogEI, Hard",   [30, 19, 29, 39, 19, 19, 27, 22, 17, 23]),
        ("RDKit k50, LogEI, Hard",    [33, 19, 35, 35, 35, 20, 37, 54, 32, 19]),
        ("RDKit k25, LogEI, Hard",    [21, 24, 31, 37, 24, 23, 26, 21, 25, 37]),
        ("RDKit k100, UCB β=2, Rand", [30, 17, 25, 36, 27, 20, 64, 27, 27, 24]),
        ("RDKit k50, UCB β=2, Rand",  [22, 19, 16, 34, 18, 18, 68, 26, 21, 28]),
        ("RDKit k25, UCB β=2, Rand",  [53, 18, np.nan, 37, 16, 17, 18, 43, 28, 15]),
        ("Validation: Optimal",       [31, 29, 48, 31, 15, 15, 73, 54, 20, 16]),
        ("Validation: Repr. only",    [45, 39, 50, 45, 28, 43, 60, 59, 49, 47]),
        ("Validation: Init. only",    [33, 19, 54, 44, 17, 35, 16, 26, 33, 47]),
        ("Validation: Random",        [np.nan]*4 + [99] + [np.nan] + [84] + [np.nan]*3),
    ]

    n_top = len(rows_top)
    n_bot = len(rows_bot)
    n_col = len(seeds)

    data_top    = np.array([[float(v) for v in r[1]] for r in rows_top])
    data_bot    = np.array([[float(v) for v in r[1]] for r in rows_bot])
    labels_top  = [r[0] for r in rows_top]
    labels_bot  = [r[0] for r in rows_bot]

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#AAAAAA")

    AMBER = "#E67E22"
    GREY  = "#AAAAAA"
    VMIN, VMAX = 10, 100

    def _cell_color_and_text(val):
        if np.isnan(val):
            return None, "FAIL", "white", True
        elif 80 <= val <= 99:
            return AMBER, str(int(val)), "white", True
        else:
            return None, str(int(val)), None, False

    def _draw_panel(ax, data, labels, panel_title, sep_after=None):
        nr, nc = data.shape
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, aspect="auto", cmap=cmap,
                       vmin=VMIN, vmax=VMAX, interpolation="nearest")

        for r in range(nr):
            for c in range(nc):
                val = data[r, c]
                fc, txt, txt_col, bold = _cell_color_and_text(val)

                if fc is not None:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=fc, zorder=2)
                    ax.add_patch(rect)
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=12, color=txt_col,
                            fontweight="bold" if bold else "normal", zorder=3)
                elif np.isnan(val):
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=12, color="white", fontweight="bold", zorder=3)
                else:
                    norm_val = (val - VMIN) / (VMAX - VMIN)
                    bg  = cmap(norm_val)
                    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                    tc  = "white" if lum < 0.5 else "black"
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=12, color=tc, zorder=3)

        if sep_after is not None:
            ax.axhline(sep_after + 0.5, color="black", lw=2.5, zorder=5)

        ax.set_yticks(range(nr))
        ax.set_yticklabels(labels, fontsize=13)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=13)
        ax.set_xlabel("Seed", fontsize=13)
        ax.set_title(panel_title, fontsize=14, fontweight="bold", pad=6)
        return im

    h_top = 1.1 * n_top + 1.5
    h_bot = 1.1 * n_bot + 1.5

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(16, h_top + h_bot + 2.5),
        gridspec_kw={"height_ratios": [n_top, n_bot], "hspace": 0.45}
    )

    im_top = _draw_panel(ax_top, data_top, labels_top,
                         "Axis 1 — Representation (selected init, LogEI)")
    _draw_panel(ax_bot, data_bot, labels_bot,
                "Axis 2 & 3 — Acquisition / Init  +  Validation",
                sep_after=5)

    cbar = fig.colorbar(im_top, ax=[ax_top, ax_bot],
                        fraction=0.018, pad=0.015, aspect=40)
    cbar.set_label("TTF (BO Iterations)", fontsize=13)
    cbar.ax.tick_params(labelsize=13)

    fig.legend(handles=[
        mpatches.Patch(facecolor=GREY,  label="FAIL — no hit within evaluation budget"),
        mpatches.Patch(facecolor=AMBER, label="Near-budget success (TTF 80–99)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.02),
       ncol=2, fontsize=13, framealpha=0.93)

    _save(fig, "SI_fig1_perseed_ttf_heatmap")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Thesis Figure Generation")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    fig5_axis1_representation()
    fig6_init_comparison()
    fig7_acqfn_comparison()
    fig9_convergence()
    SI_fig1_perseed_ttf_heatmap()

    print(f"\n{'='*60}")
    print("  Done. All figures saved to figures_out/")
    print(f"{'='*60}\n")
