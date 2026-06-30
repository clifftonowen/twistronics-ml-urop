"""CLI driver: N_m harmonic-truncation convergence study for the twisted bilayer.

WHY THIS EXISTS
---------------
The RCWA solver truncates the plane-wave basis at order N_m (passed as N=M).
The twisted bilayer costs ~(2*N_m+1)^4 plane waves, so N_m is a hard
accuracy/cost trade-off: too small silently poisons the surrogate with
under-converged spectra, too large makes the dataset unaffordable. Smaller
twist angles fold a larger moire supercell and need MORE harmonics, so the
right N_m is twist-angle dependent. This script measures that dependence and
recommends a per-twist-angle N_m policy that `generate_dataset.py` then applies.

WHAT IT DOES
------------
Part A (primary, yields the bins): one fixed "hard" geometry (large radius =
sharp dielectric features, small gap = strong interlayer coupling) evaluated at
several twist angles. At each angle it sweeps N_m and compares each spectrum to
the highest-N reference.

Part B (optional, --robustness): a few geometry variants at the smallest twist
angles, to confirm the chosen N_m is not sensitive to the other knobs.

Convergence criterion (both must hold) at truncation N:
    dCD(N)  = max_lambda |CD_N - CD_ref|          <= --tol-cd   (default 0.02)
    res(N)  = max_lambda |1 - (R + T)|            <= --tol-e    (default 1e-3)
N* is the smallest N meeting both. Smaller twist -> larger N*.

Examples
--------
Fast smoke run (validates the script end-to-end in well under a minute):
    python -m data_generation.convergence_study --smoke

Full study (minutes to tens of minutes; run in the background):
    python -m data_generation.convergence_study \
        --angles 5 8 12 20 30 --N-values 1 2 3 4 --n-lam 26

Cost/memory note: N=4 twisted is ~2.7x the channels of N=3 (which is already
~7 GB / ~180 s per full spectrum). If RAM-bound, drop --resolution to 256 or
cap --N-values at 4 and trust the trend rather than pushing to 5.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
from tqdm import tqdm

from .parameter_sampler import PARAM_NAMES, DesignParams
from .simulate_spectra import simulate_design

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "datasets", "convergence"
)

# Fixed "hard" geometry for the twist sweep (Part A): large radius -> sharp
# features, small gap -> strong evanescent coupling. Both push N_m up, so an
# N_m that converges here is conservative for easier geometries at the same twist.
HARD_GEOMETRY = {"thickness": 0.25, "gap": 0.05, "radius": 0.45}

# Geometry variants for the robustness check (Part B). Each stresses a different
# knob while staying inside parameter_sampler.BOUNDS.
ROBUSTNESS_VARIANTS = {
    "thin_membrane": {"thickness": 0.10, "gap": 0.05, "radius": 0.45},
    "wide_gap_small_hole": {"thickness": 0.25, "gap": 0.45, "radius": 0.20},
}


class TestDesign:
    """A labelled design to converge, plus which part of the study it belongs to."""

    def __init__(self, label: str, part: str, params: DesignParams):
        self.label = label
        self.part = part
        self.params = params


def build_test_designs(
    angles: list[float], robustness: bool
) -> list[TestDesign]:
    """Part A: hard geometry across twist angles. Part B (optional): geometry
    variants at the two smallest angles to confirm the bins are geometry-robust."""
    designs: list[TestDesign] = []
    for ang in angles:
        designs.append(
            TestDesign(
                f"A_theta{ang:g}",
                "A",
                DesignParams(theta_deg=float(ang), **HARD_GEOMETRY),
            )
        )
    if robustness:
        small_angles = sorted(angles)[:2]
        for ang in small_angles:
            for name, geom in ROBUSTNESS_VARIANTS.items():
                designs.append(
                    TestDesign(
                        f"B_theta{ang:g}_{name}",
                        "B",
                        DesignParams(theta_deg=float(ang), **geom),
                    )
                )
    return designs


def converged_n(
    delta_cd: dict[int, float],
    max_res: dict[int, float],
    tol_cd: float,
    tol_e: float,
) -> int | None:
    """Smallest N meeting both tolerances, or None if even the reference fails."""
    for n in sorted(delta_cd):
        if delta_cd[n] <= tol_cd and max_res[n] <= tol_e:
            return n
    return None


def recommend_policy(per_design: list[dict]) -> list[dict]:
    """Build a monotonic theta -> N_m step function from the Part-A results.

    Rule applied downstream: for a given twist theta, use the N_m of the largest
    tested angle <= theta (i.e. the harder, smaller-angle edge of its bin). We
    enforce non-increasing N_m as theta grows (smaller twist never needs fewer
    harmonics) by taking a running max from the largest angle downward, so solver
    noise can't recommend too-small an N_m for a hard angle.
    """
    part_a = sorted(
        (d for d in per_design if d["part"] == "A"),
        key=lambda d: d["params"]["theta_deg"],
    )
    # N* falls back to the highest swept N when the reference itself isn't converged.
    rows = []
    for d in part_a:
        n_star = d["n_star"]
        if n_star is None:
            n_star = max(int(n) for n in d["delta_cd"])  # reference not converged
        rows.append({"theta_min": d["params"]["theta_deg"], "N_m": int(n_star)})

    # Enforce monotonicity: walk from largest angle to smallest, ratchet N_m up.
    running = 0
    for r in reversed(rows):
        running = max(running, r["N_m"])
        r["N_m"] = running
    return rows


def run_study(
    designs: list[TestDesign],
    wavelengths: np.ndarray,
    N_values: list[int],
    a_nm: float,
    slab_material: str,
    hole_material: str,
    resolution: int,
    tol_cd: float,
    tol_e: float,
    on_design_done=None,
) -> list[dict]:
    """Simulate every (design, N), compute diagnostics vs the highest-N reference.

    `on_design_done(per_design)` is called after each design so the caller can
    persist partial results -- the N=4 runs are slow, and we don't want an
    interrupted run (OOM, Ctrl-C) to discard hours of completed designs.
    """
    n_ref = max(N_values)
    per_design: list[dict] = []

    for d in tqdm(designs, desc="designs"):
        spectra = {}
        runtimes = {}
        for n in N_values:
            t0 = time.perf_counter()
            spectra[n] = simulate_design(
                d.params,
                wavelengths,
                a_nm=a_nm,
                N=n,
                slab_material=slab_material,
                hole_material=hole_material,
                resolution=resolution,
            )
            runtimes[n] = time.perf_counter() - t0

        cd_ref = spectra[n_ref].cd
        delta_cd = {
            n: float(np.max(np.abs(spectra[n].cd - cd_ref))) for n in N_values
        }
        max_res = {
            n: float(spectra[n].energy_residual().max()) for n in N_values
        }
        n_star = converged_n(delta_cd, max_res, tol_cd, tol_e)

        per_design.append(
            {
                "label": d.label,
                "part": d.part,
                "params": d.params.as_dict(),
                "delta_cd": delta_cd,
                "max_energy_residual": max_res,
                "runtime_s": {n: round(runtimes[n], 3) for n in N_values},
                "n_star": n_star,
            }
        )
        if on_design_done is not None:
            on_design_done(per_design)
    return per_design


def write_csv(per_design: list[dict], path: str) -> None:
    fields = [
        "label", "part", "theta_deg", "thickness", "gap", "radius",
        "N", "delta_cd", "max_energy_residual", "runtime_s",
        "is_reference", "n_star",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in per_design:
            n_ref = max(int(n) for n in d["delta_cd"])
            for n in sorted(int(n) for n in d["delta_cd"]):
                w.writerow(
                    {
                        "label": d["label"],
                        "part": d["part"],
                        **{k: d["params"][k] for k in PARAM_NAMES},
                        "N": n,
                        "delta_cd": f"{d['delta_cd'][n]:.6e}",
                        "max_energy_residual": f"{d['max_energy_residual'][n]:.6e}",
                        "runtime_s": d["runtime_s"][n],
                        "is_reference": n == n_ref,
                        "n_star": d["n_star"],
                    }
                )


def write_plots(per_design: list[dict], path: str) -> str | None:
    """Overview figure: dCD-vs-N and energy-residual-vs-N for the Part-A angles.
    Returns the path written, or None if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    part_a = sorted(
        (d for d in per_design if d["part"] == "A"),
        key=lambda d: d["params"]["theta_deg"],
    )
    if not part_a:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for d in part_a:
        ns = sorted(int(n) for n in d["delta_cd"])
        label = f"theta={d['params']['theta_deg']:g} deg"
        ax1.plot(ns, [d["delta_cd"][n] for n in ns], "o-", label=label)
        ax2.semilogy(
            ns, [max(d["max_energy_residual"][n], 1e-16) for n in ns], "o-", label=label
        )
    ax1.set(xlabel="N_m", ylabel="max |CD(N) - CD(ref)|", title="CD convergence")
    ax2.set(xlabel="N_m", ylabel="max |1 - (R+T)|", title="Energy residual")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--angles", type=float, nargs="+", default=[5, 8, 12, 20, 30],
                   help="twist angles (deg) to test for convergence")
    p.add_argument("--N-values", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="harmonic truncations to sweep; the largest is the reference")
    p.add_argument("--a-nm", type=float, default=500.0, help="lattice constant in nm")
    p.add_argument("--lam-min", type=float, default=600.0, help="min wavelength nm")
    p.add_argument("--lam-max", type=float, default=850.0, help="max wavelength nm")
    p.add_argument("--n-lam", type=int, default=26, help="wavelength grid points")
    p.add_argument("--slab", default="Si3N4", help="slab material name")
    p.add_argument("--hole", default="air", help="hole-fill material name")
    p.add_argument("--resolution", type=int, default=512, help="eps-map grid resolution")
    p.add_argument("--tol-cd", type=float, default=0.02, help="max |dCD| to call converged")
    p.add_argument("--tol-e", type=float, default=1e-3, help="max energy residual to call converged")
    p.add_argument("--robustness", action="store_true",
                   help="add Part-B geometry variants at the two smallest angles")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument(
        "--smoke", action="store_true",
        help="tiny fast config (2 angles, N=[1,2], 6 wavelengths, low resolution)",
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)

    if args.smoke:
        args.angles = [args.angles[0], args.angles[-1]]
        args.N_values = [1, 2]
        args.n_lam = 6
        args.resolution = 128
        args.robustness = False

    N_values = sorted(set(int(n) for n in args.N_values))
    wavelengths = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    designs = build_test_designs(args.angles, args.robustness)

    print(
        f"Convergence study | angles={args.angles} deg | N={N_values} "
        f"(ref={N_values[-1]}) | {len(designs)} designs | "
        f"lambda {args.lam_min}-{args.lam_max} nm x{args.n_lam} | "
        f"slab={args.slab} | tol_cd={args.tol_cd} tol_e={args.tol_e:g}"
    )

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "convergence_results.csv")
    json_path = os.path.join(args.out, "convergence_summary.json")
    png_path = os.path.join(args.out, "convergence_overview.png")

    config = {
        "angles_deg": list(args.angles),
        "N_values": N_values,
        "N_ref": N_values[-1],
        "a_nm": args.a_nm,
        "wavelength_grid_nm": [args.lam_min, args.lam_max, args.n_lam],
        "slab_material": args.slab,
        "hole_material": args.hole,
        "eps_map_resolution": args.resolution,
        "tol_cd": args.tol_cd,
        "tol_e": args.tol_e,
        "hard_geometry": HARD_GEOMETRY,
        "robustness": args.robustness,
        "smoke": args.smoke,
    }

    def persist(per_design: list[dict], elapsed: float | None = None) -> None:
        """Write CSV + JSON from whatever designs have finished so far."""
        write_csv(per_design, csv_path)
        summary = {
            "config": config,
            "per_design": per_design,
            "recommended_policy": recommend_policy(per_design),
            "reference_unconverged_designs": [
                d["label"] for d in per_design if d["n_star"] is None
            ],
            "complete": elapsed is not None,
            "wall_time_s": round(elapsed, 2) if elapsed is not None else None,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    t0 = time.perf_counter()
    per_design = run_study(
        designs, wavelengths, N_values,
        a_nm=args.a_nm, slab_material=args.slab, hole_material=args.hole,
        resolution=args.resolution, tol_cd=args.tol_cd, tol_e=args.tol_e,
        on_design_done=persist,
    )
    elapsed = time.perf_counter() - t0

    persist(per_design, elapsed)
    policy = recommend_policy(per_design)
    unconverged = [d["label"] for d in per_design if d["n_star"] is None]
    plotted = write_plots(per_design, png_path)

    print(f"\nDone in {elapsed:.1f}s.")
    print("Per-angle N* (Part A):")
    for d in sorted(
        (d for d in per_design if d["part"] == "A"),
        key=lambda d: d["params"]["theta_deg"],
    ):
        star = d["n_star"] if d["n_star"] is not None else "NONE(ref unconverged)"
        print(f"  theta={d['params']['theta_deg']:>5g} deg -> N* = {star}")

    print("Recommended theta -> N_m policy (conservative, monotonic):")
    for r in policy:
        print(f"  theta >= {r['theta_min']:>5g} deg -> N_m = {r['N_m']}")

    if unconverged:
        print(
            "\nWARNING: reference N did not converge for: "
            + ", ".join(unconverged)
            + "\n  -> raise --N-values (and/or lower --resolution to fit memory)."
        )

    paths = {"csv": csv_path, "summary": json_path}
    if plotted:
        paths["plot"] = png_path
    else:
        print("(matplotlib unavailable -> skipped PNG; CSV/JSON written.)")
    for k, v in paths.items():
        print(f"  {k:8s}: {v}")
    return paths


if __name__ == "__main__":
    main()
