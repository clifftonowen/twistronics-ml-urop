"""CLI driver: N_m harmonic-truncation convergence study for the twisted bilayer.

WHY THIS EXISTS
---------------
The RCWA solver truncates the plane-wave basis at order N_m (passed as N=M).
The twisted bilayer costs ~(2*N_m+1)^4 plane waves, so N_m is a hard
accuracy/cost trade-off: too small silently poisons the surrogate with
under-converged spectra, too large makes the dataset unaffordable. Smaller
twist angles fold a larger moire supercell, and higher index contrast (a-Si)
mixes more harmonics -- both need a LARGER N_m. This script measures those
dependencies and recommends a per-twist-angle N_m policy (for the dataset's
material) that `generate_dataset.py` then applies.

CONVERGENCE TRIAGE (important)
------------------------------
For lossless real-epsilon media the S-matrix is unitary by construction, so the
energy residual |1-(R+T)| is ~machine-zero at ANY N_m -- it catches bugs, not
under-convergence. The real signal is the spectrum change vs a higher-N
reference. With the reference set to the largest swept N (N_ref), a design is:
  * CONVERGED      if some N < N_ref already matches the reference within tol
                   -> that smaller N is N* (safe to use).
  * NEEDS_HIGHER_N if only N_ref itself meets tol -> we cannot prove N_ref is
                   converged without going higher; flagged for a later (overnight)
                   N=4+ run. High contrast (a-Si) and the smallest twists land here.

Criterion at truncation N (both must hold):
    dCD(N) = max_lambda |CD_N - CD_ref| <= --tol-cd  (default 0.02)
    res(N) = max_lambda |1 - (R + T)|   <= --tol-e   (default 1e-3)

Examples
--------
Fast smoke run (validates the script end-to-end in well under a minute):
    python -m data_generation.convergence_study --smoke

Lighter material-spanning study (no N=4, no very-small twist -> low memory):
    python -m data_generation.convergence_study \
        --slabs Si3N4 TiO2 aSi --angles 8 12 20 30 --N-values 1 2 3 \
        --n-lam 9 --resolution 256

Cost/memory note: N=4 twisted is ~2.7x the channels of N=3 and was the OOM
culprit in earlier runs; this study caps at N=3 by default. Push to N>=4 only in
a dedicated, un-contended (ideally overnight) run.
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

# Geometry variants for the optional robustness check (Part B).
ROBUSTNESS_VARIANTS = {
    "thin_membrane": {"thickness": 0.10, "gap": 0.05, "radius": 0.45},
    "wide_gap_small_hole": {"thickness": 0.25, "gap": 0.45, "radius": 0.20},
}


class TestDesign:
    """A labelled (material, geometry) point to converge, tagged with its part."""

    def __init__(self, label: str, part: str, material: str, params: DesignParams):
        self.label = label
        self.part = part
        self.material = material
        self.params = params


def build_test_designs(
    angles: list[float], materials: list[str], policy_material: str, robustness: bool
) -> list[TestDesign]:
    """Part A: hard geometry across (material x twist). Part B (optional):
    geometry variants at the two smallest angles, policy-material only."""
    designs: list[TestDesign] = []
    for mat in materials:
        for ang in angles:
            designs.append(
                TestDesign(
                    f"A_{mat}_theta{ang:g}",
                    "A",
                    mat,
                    DesignParams(theta_deg=float(ang), **HARD_GEOMETRY),
                )
            )
    if robustness:
        for ang in sorted(angles)[:2]:
            for name, geom in ROBUSTNESS_VARIANTS.items():
                designs.append(
                    TestDesign(
                        f"B_{policy_material}_theta{ang:g}_{name}",
                        "B",
                        policy_material,
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


def classify(n_star: int | None, n_ref: int) -> str:
    """converged (n_star < n_ref) / needs_higher_N (n_star == n_ref) / unconverged."""
    if n_star is None:
        return "unconverged"
    if n_star < n_ref:
        return "converged"
    return "needs_higher_N"


def recommend_policy(
    per_design: list[dict], policy_material: str, n_ref: int
) -> list[dict]:
    """Build a monotonic theta -> N_m step function from the policy material's
    Part-A results.

    Rule applied downstream: for a given twist theta, use the N_m of the largest
    tested angle <= theta (the harder, smaller-angle edge of its bin). N_m is
    forced non-increasing as theta grows (smaller twist never needs fewer
    harmonics) via a running max from the largest angle down, so solver noise
    can't recommend too small an N_m. NEEDS_HIGHER_N designs contribute N_ref as
    a LOWER BOUND (real requirement may be larger -> confirm in the overnight run).
    """
    part_a = sorted(
        (d for d in per_design if d["part"] == "A" and d["material"] == policy_material),
        key=lambda d: d["params"]["theta_deg"],
    )
    rows = []
    for d in part_a:
        if d["status"] == "converged":
            n_m = int(d["n_star"])
            lower_bound = False
        else:  # needs_higher_N or unconverged -> N_ref is only a floor
            n_m = n_ref
            lower_bound = True
        rows.append(
            {
                "theta_min": d["params"]["theta_deg"],
                "N_m": n_m,
                "lower_bound": lower_bound,
            }
        )

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
    hole_material: str,
    resolution: int,
    tol_cd: float,
    tol_e: float,
    on_design_done=None,
) -> list[dict]:
    """Simulate every (design, N), compute diagnostics vs the highest-N reference.

    `on_design_done(per_design)` is called after each design so the caller can
    persist partial results -- a long run shouldn't lose finished designs if it
    is interrupted.
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
                slab_material=d.material,
                hole_material=hole_material,
                resolution=resolution,
            )
            runtimes[n] = time.perf_counter() - t0

        cd_ref = spectra[n_ref].cd
        delta_cd = {n: float(np.max(np.abs(spectra[n].cd - cd_ref))) for n in N_values}
        max_res = {n: float(spectra[n].energy_residual().max()) for n in N_values}
        n_star = converged_n(delta_cd, max_res, tol_cd, tol_e)

        per_design.append(
            {
                "label": d.label,
                "part": d.part,
                "material": d.material,
                "params": d.params.as_dict(),
                "delta_cd": delta_cd,
                "max_energy_residual": max_res,
                "runtime_s": {n: round(runtimes[n], 3) for n in N_values},
                "n_star": n_star,
                "status": classify(n_star, n_ref),
            }
        )
        if on_design_done is not None:
            on_design_done(per_design)
    return per_design


def write_csv(per_design: list[dict], path: str) -> None:
    fields = [
        "label", "part", "material", "theta_deg", "thickness", "gap", "radius",
        "N", "delta_cd", "max_energy_residual", "runtime_s",
        "is_reference", "n_star", "status",
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
                        "material": d["material"],
                        **{k: d["params"][k] for k in PARAM_NAMES},
                        "N": n,
                        "delta_cd": f"{d['delta_cd'][n]:.6e}",
                        "max_energy_residual": f"{d['max_energy_residual'][n]:.6e}",
                        "runtime_s": d["runtime_s"][n],
                        "is_reference": n == n_ref,
                        "n_star": d["n_star"],
                        "status": d["status"],
                    }
                )


def write_plots(per_design: list[dict], out_dir: str) -> list[str]:
    """One dCD-vs-N figure per material (Part A). Returns the paths written, or
    [] if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    materials = sorted({d["material"] for d in per_design if d["part"] == "A"})
    written = []
    for mat in materials:
        rows = sorted(
            (d for d in per_design if d["part"] == "A" and d["material"] == mat),
            key=lambda d: d["params"]["theta_deg"],
        )
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for d in rows:
            ns = sorted(int(n) for n in d["delta_cd"])
            ax.semilogy(
                ns,
                [max(d["delta_cd"][n], 1e-16) for n in ns],
                "o-",
                label=f"theta={d['params']['theta_deg']:g} ({d['status']})",
            )
        max_res = max(max(d["max_energy_residual"].values()) for d in rows)
        ax.set(
            xlabel="N_m",
            ylabel="max |CD(N) - CD(ref)|",
            title=f"{mat} CD convergence  (max energy res {max_res:.1e})",
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"convergence_{mat}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--slabs", nargs="+", default=["Si3N4", "TiO2", "aSi"],
                   help="slab materials to test (see materials.py)")
    p.add_argument("--policy-material", default="Si3N4",
                   help="material whose convergence defines the theta->N_m policy "
                   "used by the v1 dataset")
    p.add_argument("--angles", type=float, nargs="+", default=[8, 12, 20, 30],
                   help="twist angles (deg) to test for convergence")
    p.add_argument("--N-values", type=int, nargs="+", default=[1, 2, 3],
                   help="harmonic truncations to sweep; the largest is the reference")
    p.add_argument("--a-nm", type=float, default=500.0, help="lattice constant in nm")
    p.add_argument("--lam-min", type=float, default=600.0, help="min wavelength nm")
    p.add_argument("--lam-max", type=float, default=850.0, help="max wavelength nm")
    p.add_argument("--n-lam", type=int, default=9, help="wavelength grid points")
    p.add_argument("--hole", default="air", help="hole-fill material name")
    p.add_argument("--resolution", type=int, default=256, help="eps-map grid resolution")
    p.add_argument("--tol-cd", type=float, default=0.02, help="max |dCD| to call converged")
    p.add_argument("--tol-e", type=float, default=1e-3, help="max energy residual to call converged")
    p.add_argument("--robustness", action="store_true",
                   help="add Part-B geometry variants at the two smallest angles")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument(
        "--smoke", action="store_true",
        help="tiny fast config (1 material, 2 angles, N=[1,2], 6 wavelengths, low res)",
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)

    if args.smoke:
        args.slabs = [args.policy_material]
        args.angles = [args.angles[0], args.angles[-1]]
        args.N_values = [1, 2]
        args.n_lam = 6
        args.resolution = 128
        args.robustness = False

    N_values = sorted(set(int(n) for n in args.N_values))
    n_ref = N_values[-1]
    wavelengths = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    designs = build_test_designs(
        args.angles, args.slabs, args.policy_material, args.robustness
    )

    print(
        f"Convergence study | materials={args.slabs} (policy={args.policy_material}) "
        f"| angles={args.angles} deg | N={N_values} (ref={n_ref}) "
        f"| {len(designs)} designs | lambda {args.lam_min}-{args.lam_max} nm "
        f"x{args.n_lam} | res={args.resolution} | tol_cd={args.tol_cd}"
    )

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "convergence_results.csv")
    json_path = os.path.join(args.out, "convergence_summary.json")

    config = {
        "slabs": list(args.slabs),
        "policy_material": args.policy_material,
        "angles_deg": list(args.angles),
        "N_values": N_values,
        "N_ref": n_ref,
        "a_nm": args.a_nm,
        "wavelength_grid_nm": [args.lam_min, args.lam_max, args.n_lam],
        "hole_material": args.hole,
        "eps_map_resolution": args.resolution,
        "tol_cd": args.tol_cd,
        "tol_e": args.tol_e,
        "hard_geometry": HARD_GEOMETRY,
        "robustness": args.robustness,
        "smoke": args.smoke,
        # The policy is only validated at/above the smallest tested angle; smaller
        # twists (and any needs_higher_N material) require the overnight N>=4 run.
        "policy_validated_above_deg": float(min(args.angles)),
    }

    def persist(per_design: list[dict], elapsed: float | None = None) -> None:
        write_csv(per_design, csv_path)
        summary = {
            "config": config,
            "per_design": per_design,
            "recommended_policy": recommend_policy(
                per_design, args.policy_material, n_ref
            ),
            "needs_higher_N": [
                d["label"] for d in per_design if d["status"] != "converged"
            ],
            "complete": elapsed is not None,
            "wall_time_s": round(elapsed, 2) if elapsed is not None else None,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    t0 = time.perf_counter()
    per_design = run_study(
        designs, wavelengths, N_values,
        a_nm=args.a_nm, hole_material=args.hole, resolution=args.resolution,
        tol_cd=args.tol_cd, tol_e=args.tol_e, on_design_done=persist,
    )
    elapsed = time.perf_counter() - t0

    persist(per_design, elapsed)
    policy = recommend_policy(per_design, args.policy_material, n_ref)
    needs_higher = [d["label"] for d in per_design if d["status"] != "converged"]
    plotted = write_plots(per_design, args.out)

    print(f"\nDone in {elapsed:.1f}s.")
    print("Per (material, angle) result:")
    for d in sorted(
        (d for d in per_design if d["part"] == "A"),
        key=lambda d: (d["material"], d["params"]["theta_deg"]),
    ):
        star = d["n_star"] if d["n_star"] is not None else "-"
        print(
            f"  {d['material']:>6s} theta={d['params']['theta_deg']:>4g} deg "
            f"-> N* = {star}  [{d['status']}]"
        )

    print(f"\nRecommended theta -> N_m policy for {args.policy_material} "
          f"(validated for theta >= {min(args.angles):g} deg):")
    for r in policy:
        flag = "  (LOWER BOUND - confirm overnight)" if r["lower_bound"] else ""
        print(f"  theta >= {r['theta_min']:>4g} deg -> N_m = {r['N_m']}{flag}")

    if needs_higher:
        print(
            "\nNEEDS HIGHER-N (overnight) confirmation -- N_ref met tol but no "
            "smaller N did, so N_ref convergence is unproven:\n  "
            + ", ".join(needs_higher)
        )

    paths = {"csv": csv_path, "summary": json_path}
    for i, pth in enumerate(plotted):
        paths[f"plot{i}"] = pth
    if not plotted:
        print("(matplotlib unavailable -> skipped PNGs; CSV/JSON written.)")
    for k, v in paths.items():
        print(f"  {k:8s}: {v}")
    return paths


if __name__ == "__main__":
    main()
