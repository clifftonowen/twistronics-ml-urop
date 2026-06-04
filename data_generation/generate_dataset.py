"""CLI driver: sample designs -> simulate CD spectra -> write a dataset shard.

Examples
--------
Tiny smoke run (proves the pipeline end-to-end in seconds):
    python -m data_generation.generate_dataset --smoke

A small real shard (adjust as the parameter ranges in CLAUDE.md firm up):
    python -m data_generation.generate_dataset \
        --n 200 --N 3 --a-nm 500 --lam-min 600 --lam-max 850 --n-lam 51 \
        --seed 0 --shard shard000

Cost warning: the twisted bilayer scales as (2N+1)^4 plane waves. N=3 is ~minutes
per spectrum (CLAUDE.md section 4). Start small, run a convergence check, and
scale N only as needed for the twist/contrast regime you are sampling.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from .dataset_compiler import compile_dataset
from .parameter_sampler import BOUNDS, sample_params
from .simulate_spectra import simulate_design

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="number of designs to sample")
    p.add_argument("--N", type=int, default=3, help="harmonic truncation N_m (M=N)")
    p.add_argument("--a-nm", type=float, default=500.0, help="lattice constant in nm")
    p.add_argument("--lam-min", type=float, default=600.0, help="min wavelength nm")
    p.add_argument("--lam-max", type=float, default=850.0, help="max wavelength nm")
    p.add_argument("--n-lam", type=int, default=51, help="wavelength grid points")
    p.add_argument("--theta-inc-deg", type=float, default=0.0, help="incident polar angle")
    p.add_argument("--phi-inc-deg", type=float, default=0.0, help="incident azimuth")
    p.add_argument("--slab", default="Si3N4", help="slab material name")
    p.add_argument("--hole", default="air", help="hole-fill material name")
    p.add_argument("--resolution", type=int, default=512, help="eps-map grid resolution")
    p.add_argument("--seed", type=int, default=0, help="LHS sampling seed")
    p.add_argument("--out", default=DEFAULT_OUT, help="dataset output directory")
    p.add_argument("--shard", default="shard000", help="shard file name stem")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast config (n=4, N=1, 11 wavelengths) to validate the pipeline",
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)

    if args.smoke:
        args.n, args.N, args.n_lam = 4, 1, 11

    wavelengths = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    params = sample_params(args.n, BOUNDS, seed=args.seed)

    print(
        f"Generating {args.n} designs | N_m={args.N} | a={args.a_nm} nm | "
        f"lambda {args.lam_min}-{args.lam_max} nm x{args.n_lam} | "
        f"slab={args.slab} hole={args.hole} | incidence "
        f"({args.theta_inc_deg},{args.phi_inc_deg}) deg"
    )

    results = []
    t0 = time.perf_counter()
    for p in tqdm(params, desc="simulating designs"):
        results.append(
            simulate_design(
                p,
                wavelengths,
                a_nm=args.a_nm,
                N=args.N,
                slab_material=args.slab,
                hole_material=args.hole,
                theta_inc_deg=args.theta_inc_deg,
                phi_inc_deg=args.phi_inc_deg,
                resolution=args.resolution,
            )
        )
    elapsed = time.perf_counter() - t0

    metadata = {
        "N_m": args.N,
        "a_nm": args.a_nm,
        "wavelength_grid_nm": [args.lam_min, args.lam_max, args.n_lam],
        "incidence_deg": [args.theta_inc_deg, args.phi_inc_deg],
        "slab_material": args.slab,
        "hole_material": args.hole,
        "eps_map_resolution": args.resolution,
        "sampling_seed": args.seed,
        "param_bounds": {k: list(v) for k, v in BOUNDS.items()},
        "wall_time_s": round(elapsed, 2),
        "sec_per_spectrum": round(elapsed / max(len(results), 1), 3),
        "smoke": args.smoke,
    }

    paths = compile_dataset(params, results, args.out, metadata, shard_name=args.shard)

    max_res = max(r.energy_residual().max() for r in results)
    print(f"\nDone in {elapsed:.1f}s ({metadata['sec_per_spectrum']}s/spectrum).")
    print(f"Max energy residual |1-(R+T)| = {max_res:.2e}  (large => raise N_m).")
    for k, v in paths.items():
        print(f"  {k:10s}: {v}")
    return paths


if __name__ == "__main__":
    main()
