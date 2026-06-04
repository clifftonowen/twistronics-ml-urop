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
| 1 | N_m convergence study | Next |
| 1 | Materials + incident angle as input dimensions | Planned |
| 2 | Forward surrogate | Not started |
| 3 | Inverse design (tandem / generative) | Not started |

The pipeline passes its physics sanity checks: energy is conserved to machine
precision for lossless media, an untwisted (achiral) bilayer returns CD ≈ 0, and
introducing a twist turns CD on. See `scripts/check_setup.py`.

## Layout

```
.
├── data_generation/          # the stage-1 pipeline (a Python package)
│   ├── paths.py              # puts the rcwa4d submodule on sys.path
│   ├── materials.py          # relative permittivities (Si3N4, a-Si, TiO2, SiO2)
│   ├── geometry.py           # builds the unit-cell permittivity map
│   ├── parameter_sampler.py  # Latin-hypercube sampling of design parameters
│   ├── simulate_spectra.py   # drives RCWA-4D → T_RCP, T_LCP, CD
│   ├── dataset_compiler.py   # stacks results into X/Y tensors + scalers + metadata
│   └── generate_dataset.py   # command-line entry point
├── scripts/
│   └── check_setup.py        # environment + solver health check
├── datasets/                 # generated data (git-ignored)
│   ├── raw/                  # full solver outputs, kept for reprocessing
│   └── processed/            # normalized tensors + scalers + metadata
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
python -m data_generation.generate_dataset `
    --n 200 --N 3 --a-nm 500 `
    --lam-min 600 --lam-max 850 --n-lam 51 `
    --seed 0 --shard shard000
```

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
- **Materials** are non-dispersive constants in v1. True `n(λ)` dispersion is a
  planned refinement (it requires rebuilding the permittivity map per wavelength).
- **Convergence.** The harmonic truncation `N_m` controls accuracy and cost; the
  twisted bilayer scales as `(2N_m + 1)^4`. Run a convergence study before trusting
  a batch — under-converged spectra poison the surrogate without any obvious symptom.

## Roadmap

Near term, to finish a trustworthy stage-1 dataset:

- **Convergence study** across `N_m` to pick the smallest accurate truncation for the
  twist-angle range we sample.
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
