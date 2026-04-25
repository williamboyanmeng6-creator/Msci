# Bayesian Optimisation for Molecular Property Prediction

MSci Thesis — William Meng, Imperial College London (2025–2026)

This repository contains the code for my MSci thesis, which investigates Bayesian
Optimisation (BO) as a strategy for efficiently screening a large molecular dataset
for high redox potential (E₀ vs. SHE). Rather than evaluating all ~2600 candidates,
a GP surrogate model is fitted to a small initial set and then used to suggest which
molecule to evaluate next, guided by an acquisition function.

The experiments are structured as a three-axis ablation study: representation,
acquisition function, and initialisation strategy. Each axis isolates one design
choice while holding everything else constant.

---

## Repository Structure

```
thesis-repo/
├── src/
│   ├── data/
│   │   ├── rdkit_feature_generation.py      # RDKit descriptors, PCA, selected-init pool
│   │   └── morgan_rf_dataset_generation.py  # Morgan fingerprint feature sets
│   ├── models/
│   │   ├── bo_rdkit_matern.py               # Axis 1: RDKit + Matérn (incl. PCA variants)
│   │   ├── bo_morgan_matern.py              # Axis 1: Morgan + Matérn (kernel-mismatch)
│   │   ├── bo_tanimoto_morgan.py            # Axis 1: Morgan + TanimotoGP (best)
│   │   ├── bo_acquisition_comparison.py     # Axis 2: LogEI vs. UCB
│   │   ├── bo_ucb_beta_sweep.py             # Axis 2: UCB β sensitivity sweep
│   │   ├── bo_initialisation_comparison.py  # Axis 3: hard-init vs. selected-init
│   │   ├── bo_optimal_validation.py         # Validation: optimal config ablation
│   │   └── gp_calibration.py               # GP uncertainty calibration study
│   └── hpc/
│       └── bo_ninit_variation.py            # n_init sweep (designed for HPC)
├── scripts/
│   └── generate_figures.py                  # Produces all thesis figures from outputs
├── data/                                    # Datasets (not tracked by git)
├── figures_out/                             # Generated figures
├── environment.yml
└── README.md
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate msci_bo
```

For HPC (Imperial cx3), see the setup notes in `src/hpc/bo_ninit_variation.py`.

---

## Running the Experiments

The scripts should be run in the order below. Steps 1–2 produce the feature CSVs
that all subsequent BO scripts depend on.

### Step 1 — Generate RDKit feature sets and preprocessing

```bash
python src/data/rdkit_feature_generation.py
```

Computes all ~200 RDKit physicochemical descriptors, applies RF importance-mass
threshold selection at τ ∈ {0.7, 0.8, 0.9} (giving k101, k44, k29 feature sets),
generates PCA variants (2D, 90%, 95% variance), and saves the 12-molecule
selected-initialisation pool. These feature sets are used by the Axis 2/3 scripts.

### Step 2 — Generate Morgan fingerprint feature sets

```bash
python src/data/morgan_rf_dataset_generation.py
```

Computes ECFP4 fingerprints (4096-bit combined oxidised+reduced), ranks bits by RF
importance, and saves three percentile-thresholded feature sets (T=0.70/0.80/0.90).
The top-k sets used in Axis 1 are separate; see the note in that script.

---

### Axis 1 — Effect of Representation

Tests whether the choice of molecular representation and kernel pairing matters.

```bash
# RDKit top-k descriptors + Matérn-2.5 ARD (also runs PCA variants)
python src/models/bo_rdkit_matern.py

# Morgan fingerprints + Matérn (the kernel-mismatch comparison)
python src/models/bo_morgan_matern.py

# Morgan fingerprints + TanimotoGP (best-performing configuration)
python src/models/bo_tanimoto_morgan.py
```

All three use the same selected-initialisation (12 fixed molecules), LogEI, and
100-iteration budget, so the only difference is representation + kernel.

**Key finding:** TanimotoGP + Morgan k100 achieves median TTF = 23, compared to
27 for RDKit k100 + Matérn. Morgan k25 + Matérn is 0/10 seeds — too sparse for
Euclidean-distance-based kernel.

---

### Axis 2 — Effect of Acquisition Function

```bash
# LogEI vs. UCB (β=2.0) on RF-thresholded RDKit feature sets, random init
python src/models/bo_acquisition_comparison.py

# UCB β sensitivity sweep (β ∈ {0.1, 0.5, 1.0, 1.5, 2.0}), hard-init
python src/models/bo_ucb_beta_sweep.py
```

**Key finding:** LogEI and UCB (β=1.0) perform comparably on k100 features.
UCB β ∈ {0.5, 1.0, 1.5, 2.0} are roughly stable; β=0.1 is too greedy.

---

### Axis 3 — Effect of Initialisation Strategy

```bash
# Hard-init (random from bottom-90%) vs. selected-init (curated 12)
# Runs both RDKit+Matérn and TanimotoGP+Morgan arms
python src/models/bo_initialisation_comparison.py
```

**Key finding:** Hard-init consistently outperforms selected-init. The likely
reason is that hard-init still draws from ~2380 molecules (wider chemical
space), while selected-init is constrained to 12 hand-picked structures.

---

### Validation Study

```bash
# Single-factor ablation: optimal config vs. each design choice swapped
python src/models/bo_optimal_validation.py
```

Four conditions: (1) optimal (TanimotoGP + Morgan100 + hard-init), (2) RDKit
k101 + Matérn + hard-init, (3) TanimotoGP + Morgan100 + random-init, (4) random
baseline. Budget extended to 115 iterations.

---

### GP Calibration

```bash
python src/models/gp_calibration.py
```

80/20 train-test split, repeated for 10 seeds. Checks R², RMSE, and empirical
coverage at credible levels {50%, 68%, 80%, 90%, 95%}. Both the Matérn k101 and
TanimotoGP Morgan100 surrogates are overconfident (spread ratio ≈ 0.5), meaning
their predicted uncertainties are roughly half as wide as the actual errors.

---

### n_init Variation (HPC)

```bash
# Single n_init value (PBS array task):
python src/hpc/bo_ninit_variation.py --n_init 12 --data_dir data/ --out_dir results/ninit/

# All values locally (slow):
python src/hpc/bo_ninit_variation.py --run_all --data_dir data/ --out_dir results/ninit/
```

---

### Generate Thesis Figures

```bash
python scripts/generate_figures.py
```

Reads all experiment output CSVs and writes PDFs to `figures_out/`. Requires
all experiments to have been run first.

---

## Key Design Decisions

**Why TanimotoGP for Morgan fingerprints?**
Morgan bits are binary and the Tanimoto (Jaccard) coefficient is the standard
fingerprint similarity metric in cheminformatics. Using Matérn on binary vectors
treats bit co-absence as evidence of similarity, which is incorrect — Tanimoto
counts only shared active bits in the numerator. In practice this shows as
0/10 seeds reaching the target for Morgan k25 + Matérn.

**Why hard-init over selected-init?**
Starting from the bottom of the E0 distribution forces the GP to learn a steep
gradient toward the unexplored high-value tail. The curated selected-init,
while chemically diverse, turns out not to help — it's too constrained to 12
fixed structures.

**Why LogEI over vanilla EI?**
LogEI is numerically stable in low-improvement regimes where standard EI can
underflow to zero, preventing useful exploration near the optimum.

**Why are the GPs overconfident?**
The GP marginal likelihood tends to push the noise variance down on medium-sized
datasets, making the model "explain" the data almost perfectly and underestimate
uncertainty in test regions. This is a well-known behaviour; it doesn't break BO
but means the posterior uncertainty shouldn't be taken literally.

---

## Dependencies

See `environment.yml`. Core packages: `numpy`, `pandas`, `matplotlib`, `scipy`,
`scikit-learn`, `rdkit`, `torch`, `botorch`, `gpytorch`.

---

## Data

The molecular dataset (`Ritsuki_full_data_with_rdkit_descriptors.csv`) is not
included in this repository due to size. Please contact the project supervisor
or refer to the thesis for data provenance and access instructions.
