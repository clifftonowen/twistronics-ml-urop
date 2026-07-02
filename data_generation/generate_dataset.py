"""CLI driver: sample designs -> simulate CD spectra -> write a dataset shard.

Examples
--------
Tiny smoke run (proves the pipeline end-to-end in seconds):
    python -m data_generation.generate_dataset --smoke

A small real shard (adjust as the parameter ranges firm up):
    python -m data_generation.generate_dataset \
        --n 200 --N 3 --a-nm 500 --lam-min 600 --lam-max 850 --n-lam 51 \
        --seed 0 --shard shard000

Cost warning: the twisted bilayer scales as (2N+1)^4 plane waves. N=3 is ~minutes
per spectrum. Start small, run a convergence check, and
scale N only as needed for the twist/contrast regime you are sampling.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from .convergence import N_M_POLICY, n_m_for_twist
from .dataset_compiler import compile_dataset
from .parameter_sampler import BOUNDS, DesignParams, sample_params
from .simulate_spectra import simulate_design

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def _simulate_job(job: dict):
    """Top-level (picklable) worker: unpack one job dict -> SpectrumResult.

    Each worker process builds its own `rcwa` object; nothing solver-side is
    shared across processes. Kept at module scope so `ProcessPoolExecutor` can
    pickle it on Windows (spawn start method re-imports this module).
    """
    return simulate_design(
        DesignParams(**job["params"]),
        job["wavelengths"],
        a_nm=job["a_nm"],
        N=job["n_m"],
        slab_material=job["slab"],
        hole_material=job["hole"],
        theta_inc_deg=job["theta_inc_deg"],
        phi_inc_deg=job["phi_inc_deg"],
        resolution=job["resolution"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="number of designs to sample")
    p.add_argument(
        "--N", type=int, default=None,
        help="fixed harmonic truncation N_m (M=N) for ALL designs; overrides the "
        "adaptive per-twist policy. Default: adaptive (see --no-adaptive-nm).",
    )
    p.add_argument(
        "--no-adaptive-nm", dest="adaptive_nm", action="store_false",
        help="disable the per-twist N_m policy; use a single N_m (--N, else 3)",
    )
    p.set_defaults(adaptive_nm=True)
    p.add_argument("--a-nm", type=float, default=500.0, help="lattice constant in nm")
    p.add_argument("--lam-min", type=float, default=600.0, help="min wavelength nm")
    p.add_argument("--lam-max", type=float, default=850.0, help="max wavelength nm")
    p.add_argument("--n-lam", type=int, default=51, help="wavelength/frequency grid points")
    p.add_argument(
        "--f-min", type=float, default=None,
        help="min normalized frequency f=a/lambda. Give WITH --f-max to sample "
        "directly on the frequency axis instead of --lam-min/--lam-max. Since "
        "v1 materials are non-dispersive, the solver output depends only on f "
        "(and angle) -- not on a_nm separately -- so a wide f grid lets a_nm be "
        "chosen (or re-chosen) post-hoc to land features in 600-850 nm at zero "
        "extra solver cost (see README 'Conventions').",
    )
    p.add_argument(
        "--f-max", type=float, default=None,
        help="max normalized frequency f=a/lambda; see --f-min",
    )
    p.add_argument("--theta-inc-deg", type=float, default=0.0, help="incident polar angle")
    p.add_argument("--phi-inc-deg", type=float, default=0.0, help="incident azimuth")
    p.add_argument("--slab", default="Si3N4", help="slab material name")
    p.add_argument("--hole", default="air", help="hole-fill material name")
    p.add_argument("--resolution", type=int, default=512, help="eps-map grid resolution")
    p.add_argument("--seed", type=int, default=0, help="LHS sampling seed")
    p.add_argument("--out", default=DEFAULT_OUT, help="dataset output directory")
    p.add_argument("--shard", default="shard000", help="shard file name stem")
    p.add_argument(
        "--workers", type=int, default=1,
        help="parallel worker processes for simulating designs (default 1 = "
        "serial). Each worker builds its own rcwa object; mind memory at "
        "high N_m (N=3 twisted is ~7 GB/spectrum, so keep workers small there).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast config (n=4, N=1, 11 wavelengths) to validate the pipeline",
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, str]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.smoke:
        # Force a fast fixed truncation; smoke validates plumbing, not convergence.
        args.n, args.N, args.n_lam = 4, 1, 11

    if (args.f_min is None) != (args.f_max is None):
        parser.error("--f-min and --f-max must be given together")
    if args.f_min is not None:
        # Sample directly on the (a_nm-independent) frequency axis; wavelengths
        # are just the a_nm label attached to this shard, freely re-pickable
        # later (see --f-min help / README).
        freqs_target = np.linspace(args.f_min, args.f_max, args.n_lam)
        wavelengths = args.a_nm / freqs_target
    else:
        wavelengths = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    params = sample_params(args.n, BOUNDS, seed=args.seed)

    # Decide N_m per design. Priority: explicit --N (fixed for all) > adaptive
    # per-twist policy > legacy fixed default (3).
    if args.N is not None:
        n_m_per_design = [args.N] * len(params)
        nm_desc = f"N_m={args.N} (fixed)"
    elif args.adaptive_nm:
        n_m_per_design = [n_m_for_twist(p.theta_deg) for p in params]
        nm_desc = f"N_m=adaptive {sorted(set(n_m_per_design))}"
    else:
        n_m_per_design = [3] * len(params)
        nm_desc = "N_m=3 (fixed default)"

    print(
        f"Generating {args.n} designs | {nm_desc} | a={args.a_nm} nm | "
        f"lambda {wavelengths.min():.1f}-{wavelengths.max():.1f} nm x{args.n_lam} | "
        f"slab={args.slab} hole={args.hole} | incidence "
        f"({args.theta_inc_deg},{args.phi_inc_deg}) deg"
    )

    t0 = time.perf_counter()
    if args.workers > 1:
        jobs = [
            {
                "params": p.as_dict(),
                "n_m": n_m,
                "wavelengths": wavelengths,
                "a_nm": args.a_nm,
                "slab": args.slab,
                "hole": args.hole,
                "theta_inc_deg": args.theta_inc_deg,
                "phi_inc_deg": args.phi_inc_deg,
                "resolution": args.resolution,
            }
            for p, n_m in zip(params, n_m_per_design)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(
                tqdm(
                    ex.map(_simulate_job, jobs),
                    total=len(jobs),
                    desc=f"simulating designs ({args.workers} workers)",
                )
            )
    else:
        results = []
        for p, n_m in zip(tqdm(params, desc="simulating designs"), n_m_per_design):
            results.append(
                simulate_design(
                    p,
                    wavelengths,
                    a_nm=args.a_nm,
                    N=n_m,
                    slab_material=args.slab,
                    hole_material=args.hole,
                    theta_inc_deg=args.theta_inc_deg,
                    phi_inc_deg=args.phi_inc_deg,
                    resolution=args.resolution,
                )
            )
    elapsed = time.perf_counter() - t0

    adaptive_used = args.N is None and args.adaptive_nm
    unique_nm = sorted(set(n_m_per_design))
    metadata = {
        # a single int when every design used the same N_m (the v1 flat case),
        # else "adaptive" -- the per-design list and policy carry the detail.
        "N_m": unique_nm[0] if len(unique_nm) == 1 else "adaptive",
        "N_m_per_design": n_m_per_design,
        "N_m_policy": [list(b) for b in N_M_POLICY] if adaptive_used else None,
        "a_nm": args.a_nm,
        # freq_grid is the canonical, a_nm-independent axis (f = a/lambda);
        # wavelength_grid_nm is just that axis labelled at this shard's a_nm
        # and is only a free post-hoc relabeling while materials stay
        # non-dispersive (see README "Conventions" / --f-min help).
        "freq_grid": [float(results[0].freqs.min()), float(results[0].freqs.max()), args.n_lam],
        "wavelength_grid_nm": [float(wavelengths.min()), float(wavelengths.max()), args.n_lam],
        "sampled_on": "frequency" if args.f_min is not None else "wavelength_nm",
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
