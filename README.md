# Twistronics: ML-Driven Multiband Polarization Converter

Inverse-design framework for **twisted bilayer photonic crystals (TBPCs)** that act
as multiband polarization converters. The goal is a transmissive device with
strong, sharp circular dichroism (CD) in the **600–850 nm** band across a range of
incident angles.

Rigorous electromagnetic simulation of moiré supercells is expensive, so the plan
is three-staged:

1. **Generate** a dataset of (structure → optical response) pairs with the RCWA-4D
   solver.
2. **Train a forward surrogate** that predicts the CD spectrum from structural
   parameters in sub-second time.
3. **Train an inverse/generative model** that maps a *target* CD spectrum back to
   physical design parameters, with every proposal re-verified in RCWA-4D.

This repository currently covers stage 1.

## Status

| Stage | Component | State |
|-------|-----------|-------|
| 1 | Environment + solver import bridge | Done |
| 1 | Parameter sampling (θ, t, gap, r) | Done |
| 1 | RCWA driver → CD spectra | Done |
| 1 | Dataset compiler (tensors, scalers, metadata) | Done |
| 1 | N_m convergence study (Si₃N₄, TiO₂) | Done — see findings below |
| 1 | Per-design N_m policy + adaptive hook | Done (v1 = flat N=2) |
| 1 | Stage-1 walkthrough notebook | Done |
| 1 | v1 dataset (fixed N=2, approximate) | Next |
| 1 | a-Si convergence + N≥3 confirmation | Deferred (needs faster compute) |
| 1 | Materials + incident angle as input dimensions | Planned |
| 2 | Forward surrogate | Not started |
| 3 | Inverse design (tandem / generative) | Not started |

The pipeline passes its physics sanity checks: energy is conserved to machine
precision for lossless media, an untwisted (achiral) bilayer returns CD ≈ 0, and
introducing a twist turns CD on. See `scripts/check_setup.py`.

### N_m convergence study — findings

`data_generation/convergence_study.py` sweeps the harmonic truncation `N_m` over
representative (material, twist) points and compares each spectrum to the
highest-`N` reference. Two results shape the dataset plan:

- **Cost wall.** On current hardware a twisted spectrum costs ~3–11 s at `N=1`,
  ~1.5 min at `N=2`, and **~33 min at `N=3`** (cost ∝ `(2N_m+1)⁴`). A converged
  (`N≥3`) dataset is multi-day and not feasible yet; **`N=2` is the practical
  ceiling** for a sizeable dataset.
- **No clean twist trend (8–30°).** Si₃N₄ convergence is not monotonic in twist:
  the absolute CD error is dominated by CD *magnitude* (high-twist designs have
  larger CD), so binning `N_m` by twist is unsupported. **v1 uses a flat `N=2`.**

So the v1 dataset is **approximate ("v0")**: accurate at low twist, but
under-resolved (CD off by ~0.05–0.07) at high twist and for higher-contrast
materials. It is meant to stand up the forward surrogate, not to report final
device numbers. A converged dataset awaits faster compute (GPU/cluster) or a
solver-usage speedup.

## Layout

```
.
├── data_generation/          # the stage-1 pipeline (a Python package)
│   ├── paths.py              # puts the rcwa4d submodule on sys.path
│   ├── materials.py          # relative permittivities (Si3N4, a-Si, TiO2, SiO2)
│   ├── geometry.py           # builds the unit-cell permittivity map
│   ├── parameter_sampler.py  # Latin-hypercube sampling of design parameters
│   ├── simulate_spectra.py   # drives RCWA-4D → T_RCP, T_LCP, CD
│   ├── convergence_study.py  # N_m convergence study (material × twist sweep)
│   ├── convergence.py        # per-design N_m policy (v1 = flat N=2)
│   ├── dataset_compiler.py   # stacks results into X/Y tensors + scalers + metadata
│   └── generate_dataset.py   # command-line entry point
├── notebooks/
│   └── 01_data_exploration.ipynb  # narrated Stage-1 walkthrough
├── scripts/
│   └── check_setup.py        # environment + solver health check
├── datasets/                 # generated data (git-ignored)
│   ├── raw/                  # full solver outputs, kept for reprocessing
│   ├── processed/            # normalized tensors + scalers + metadata
│   └── convergence/          # convergence-study CSV / JSON / plots
├── rcwa4d/                   # git submodule: fork of fancompute/rcwa4d
│   └── rcwa4d/               # solver package (utils.py, expansion4d.py)
├── requirements.txt
└── README.md
```

## Setup

The solver lives in a git submodule, so clone with submodules (or initialize them
after the fact):

```powershell
git clone --recurse-submodules <repo-url>
# or, in an existing clone:
git submodule update --init --recursive
```

Create the virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirm everything imports and the solver behaves:

```powershell
python -m scripts.check_setup
```

A passing run reports energy conservation, CD ≈ 0 for the untwisted bilayer, and
nonzero CD once twisted.

## Generating data

A fast end-to-end smoke run (a few designs, coarse grid, takes under a minute):

```powershell
python -m data_generation.generate_dataset --smoke
```

A real shard, with parameters spelled out:

```powershell
# v1 (approximate) shard: fixed N=2, reduced grid -- ~8-12 h, fits an overnight run.
# (N=3 would be accurate but ~days on current hardware; see the findings above.)
python -m data_generation.generate_dataset `
    --n 120 --N 2 --a-nm 500 `
    --lam-min 600 --lam-max 850 --n-lam 26 `
    --seed 0 --shard v0_n120
```

`N_m` is chosen per design by `data_generation/convergence.py` (v1 = flat `N=2`);
pass `--N` to override it with a single fixed truncation, as above. Pass
`--workers K` to simulate designs in `K` parallel processes (near-linear
speedup at N=2; mind memory at N=3, ~7 GB/spectrum, so keep `K` small there).

Because v1 materials are **non-dispersive**, the solver output depends only on
the normalized frequency `f = a/λ` (and incident angle) — not on `a_nm`
separately. So `--lam-min/--lam-max` (which fix an `f` range *for the given
`--a-nm`*) are really just a convenience; pass `--f-min/--f-max` instead to
sample a wide, `a_nm`-independent frequency band directly. `a_nm` then becomes
a **free, zero-cost, post-hoc choice** at analysis time — re-labelling
`λ = a_nm / f` on already-simulated data — which is how the "lattice-constant
scan to land sharp CD features in 600–850 nm" roadmap item gets done without
running the solver again. This trick stops working once dispersive `n(λ)` is
introduced (then a_nm and λ are coupled through the material itself).

Outputs land in `datasets/`:

- `raw/<shard>.npz` — wavelengths, frequencies, T/R for both handedness, CD, and the
  per-wavelength energy residual.
- `processed/<shard>.npz` — standardized inputs `X` and the CD target `Y`.
- `processed/<shard>_xscaler.npz` — the feature scaler, so inference and the
  verification loop apply the identical transform.
- `processed/<shard>_metadata.json` — full provenance (N_m, grids, materials,
  lattice constant, seed, bounds, timing).

## Conventions

These are easy to get silently wrong, so they are fixed project-wide and recorded
in every shard's metadata.

- **Circular dichroism** is defined in transmission,
  `CD = (T_RCP − T_LCP) / (T_RCP + T_LCP)`, and lies in [−1, 1].
- **Units.** All lengths (thickness, gap, hole radius) are expressed as fractions of
  the lattice constant `a`; the solver runs with `a = 1`. Physical wavelength enters
  only through the normalized frequency `f = a / λ`, so the lattice constant in
  nanometres (`a_nm`) is what places resonances inside the 600–850 nm window.
  **`f` is the canonical, dataset-portable axis** (stored as `freqs` in every raw
  shard); `wavelengths_nm` is just that axis labelled at the shard's `a_nm`. While
  materials stay non-dispersive, `a_nm` can be freely re-chosen after the fact —
  see "Generating data" above.
- **Materials** are non-dispersive constants in v1. True `n(λ)` dispersion is a
  planned refinement (it requires rebuilding the permittivity map per wavelength).
- **Convergence.** The harmonic truncation `N_m` controls accuracy and cost; the
  twisted bilayer scales as `(2N_m + 1)^4`. Judge convergence by the **spectrum
  change vs a higher-`N` reference** (`max_λ |CD_N − CD_ref|`), not by the energy
  residual: for lossless real-`ε` media the solver's scattering matrix is unitary
  by construction, so `|1 − (R+T)|` stays ~machine-zero at *any* `N_m` — it catches
  bugs, not under-convergence. v1 uses a cost-driven flat `N=2` (see Status).

## Roadmap

Near term, to finish a trustworthy stage-1 dataset:

- **Converged dataset.** The v1 dataset is approximate (flat `N=2`, cost-driven).
  Reaching a converged (`N≥3`) dataset — and confirming a-Si — needs faster compute
  (GPU/cluster) or a solver-usage speedup (the twisted path re-expands the
  plane-wave basis every wavelength, a likely optimization target).
- **Lattice-constant scan** to land sharp CD features in 600–850 nm.
- **Extend the design space** so the dataset matches the planned model inputs:
  - materials (Si₃N₄, a-Si, TiO₂) as a categorical input dimension;
  - incident angle as a continuous input, supporting the angle-robustness goal.

Later stages:

- **Forward surrogate** (stage 2): an MLP baseline mapping parameters to the CD
  spectrum, validated against held-out structures.
- **Inverse design** (stage 3): a tandem network (inverse model feeding a frozen
  forward model) as the first approach to the one-to-many problem, with generative
  alternatives and gradient-based refinement as follow-ups. Every proposed design is
  re-simulated in RCWA-4D before any metric is reported.

## References

The solver is a fork of [fancompute/rcwa4d](https://github.com/fancompute/rcwa4d);
its example notebooks document the API used here. The method and its convergence
behaviour are described in Lou & Fan, *Comput. Phys. Commun.* 306 (2025) 109356, and
the underlying theory in Lou et al., *Phys. Rev. Lett.* 126, 136101 (2021).
