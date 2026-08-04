"""BLOCKING gate for the peak-centred campaign: can a cheap N=2 pass find the
same CD peak that N=3 does?

The whole cost model rests on this. Pass 1 must be dense enough to find the
right lobe (the old 26-point grid picks the wrong one for 7 of 20 designs), and
dense scans are only affordable at reduced truncation -- ~2 s/wavelength at N=2
against ~34 s/wavelength at N=3 on this hardware. But resonance positions shift
with the harmonic truncation N_m, and nothing so far has measured by how much.
If they shift by more than the window can absorb, pass 1 has to run at N=3 and
the per-design cost rises roughly 17x, which is a scope change rather than a
tuning knob.

Method: re-run pass 1 at N=2 on designs whose N=3 peak positions are already
known from the dense rescans on disk, and compare. No new N=3 simulation is
needed -- the reference answers already exist.

Accept: at least `--min-pass` of the designs land within `--tol` (relative, in
f) of the known N=3 peak. The default tolerance is half the window half-width,
so a passing design's true peak sits comfortably inside pass 2's window rather
than near its edge.

Usage:
    python -m scripts.locate_gate --device gpu
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import data_generation  # noqa: F401 -- ensures rcwa4d is on sys.path
import rcwa4d.backend as backend
from data_generation.parameter_sampler import PARAM_NAMES, DesignParams
from data_generation.peak_passes import (
    DEFAULT_LOCATE_F_MAX,
    DEFAULT_LOCATE_F_MIN,
    DEFAULT_LOCATE_PTS,
    locate_pass,
)
from data_generation.peak_window import DEFAULT_HALF_WIDTH, locate_peak

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")
ARTIFACTS = os.path.join(ROOT, "models", "artifacts")

DENSE_SHARDS = [
    "dense_rescan_highcd_n3_gpu_merged60",
    "dense_rescan_highcd_n3_gpu_merged60_pilot17",
]
# Designs inside the new theta 25-30 box that already have an N=3 dense scan.
# Chosen to span the peak-frequency range (0.63-0.81) and to include design 10,
# which carries six comparable lobes -- the hardest case for any peak locator,
# and the one most likely to swap lobes between truncations.
GATE_INDICES = [38, 36, 15, 14, 10]


def load_reference_peaks(a_nm: float) -> dict[int, dict]:
    """N=3 peak positions from the dense rescans already on disk."""
    ref: dict[int, dict] = {}
    for shard in DENSE_SHARDS:
        path = os.path.join(RAW, f"{shard}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        freqs = a_nm / d["dense_wavelengths_nm"]
        for idx in d["indices"]:
            loc = locate_peak(freqs, d[f"design{idx}_dense_cd"])
            ref[int(idx)] = {
                "f_peak": loc.f_peak,
                "peak_abs_cd": loc.peak_abs_cd,
                "n_lobes": loc.n_competing_lobes,
                "grid_step_f": loc.grid_step_f,
                "shard": shard,
            }
    return ref


def main(argv: list[str] | None = None) -> bool:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indices", type=int, nargs="+", default=GATE_INDICES)
    p.add_argument("--source-shard", default="highcd_n3_gpu_merged60", help="shard providing X")
    p.add_argument("--locate-N", type=int, default=2)
    p.add_argument("--f-min", type=float, default=DEFAULT_LOCATE_F_MIN)
    p.add_argument("--f-max", type=float, default=DEFAULT_LOCATE_F_MAX)
    p.add_argument("--n-pts", type=int, default=DEFAULT_LOCATE_PTS)
    p.add_argument("--a-nm", type=float, default=500.0)
    p.add_argument("--slab", default="Si3N4")
    p.add_argument("--hole", default="air")
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument(
        "--tol", type=float, default=DEFAULT_HALF_WIDTH / 2,
        help="max |f_peak(N=2) - f_peak(N=3)| / f_peak(N=3) to count as a hit",
    )
    p.add_argument("--min-pass", type=int, default=4, help="hits required to pass the gate")
    p.add_argument("--out", default=os.path.join(ARTIFACTS, "locate_gate.json"))
    args = p.parse_args(argv)

    backend.set_device(args.device)

    X = np.load(os.path.join(RAW, f"{args.source_shard}.npz"))["X"]
    ref = load_reference_peaks(args.a_nm)
    missing = [i for i in args.indices if i not in ref]
    if missing:
        raise SystemExit(f"no N=3 reference peak on disk for design(s) {missing}")

    print(
        f"Locate gate: N={args.locate_N}, f in [{args.f_min}, {args.f_max}] x {args.n_pts} pts "
        f"(step {(args.f_max - args.f_min) / (args.n_pts - 1):.4f}), device={args.device}\n"
        f"Accept: >= {args.min_pass}/{len(args.indices)} within {args.tol:.1%} of the N=3 peak\n"
    )

    rows = []
    for idx in args.indices:
        params = DesignParams(**dict(zip(PARAM_NAMES, X[idx])))
        t0 = time.perf_counter()
        got = locate_pass(
            params, a_nm=args.a_nm, f_min=args.f_min, f_max=args.f_max,
            n_pts=args.n_pts, N=args.locate_N,
            slab_material=args.slab, hole_material=args.hole,
        )
        elapsed = time.perf_counter() - t0
        r = ref[idx]
        rel_err = abs(got.location.f_peak - r["f_peak"]) / r["f_peak"]
        hit = rel_err <= args.tol
        rows.append({
            "index": int(idx),
            "params": {k: float(v) for k, v in zip(PARAM_NAMES, X[idx])},
            "f_peak_locate": got.location.f_peak,
            "f_peak_ref_n3": r["f_peak"],
            "rel_err": float(rel_err),
            "hit": bool(hit),
            "peak_abs_cd_locate": got.location.peak_abs_cd,
            "peak_abs_cd_ref_n3": r["peak_abs_cd"],
            "n_lobes_locate": got.location.n_competing_lobes,
            "n_lobes_ref_n3": r["n_lobes"],
            "q_estimate_locate": float(got.location.q_estimate),
            "energy_residual_max": float(got.spectrum.energy_residual().max()),
            "seconds": round(elapsed, 1),
        })
        print(
            f"  idx={idx:3d} theta={X[idx][0]:5.1f}  "
            f"f_peak: N=2 {got.location.f_peak:.4f} vs N=3 {r['f_peak']:.4f}  "
            f"rel_err={rel_err:7.2%}  {'HIT ' if hit else 'MISS'}  "
            f"| peak|CD| {got.location.peak_abs_cd:.3f} vs {r['peak_abs_cd']:.3f}  "
            f"| lobes {got.location.n_competing_lobes} vs {r['n_lobes']}  "
            f"| {elapsed:.0f}s"
        )

    n_hit = sum(r["hit"] for r in rows)
    passed = n_hit >= args.min_pass
    per_design = float(np.mean([r["seconds"] for r in rows]))

    print(f"\n  {n_hit}/{len(rows)} within {args.tol:.1%}  ->  GATE {'PASSED' if passed else 'FAILED'}")
    print(f"  locate pass measured at {per_design:.0f} s/design ({per_design / args.n_pts:.2f} s/wavelength)")
    if not passed:
        print(
            "\n  Do NOT proceed to the campaign. A miss means the cheap pass centres the\n"
            "  window on a feature N=3 does not agree with, so pass 2 would resolve the\n"
            "  wrong resonance. Options: raise --locate-N (cost per design rises ~17x at\n"
            "  N=3), or widen the pass-2 window so a mislocated centre still contains the\n"
            "  true peak (which costs resolution -- the thing this campaign exists to buy)."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "passed": passed, "n_hit": n_hit, "n_total": len(rows),
            "tol": args.tol, "min_pass": args.min_pass,
            "locate_N": args.locate_N, "f_min": args.f_min, "f_max": args.f_max,
            "n_pts": args.n_pts, "device": args.device,
            "seconds_per_design": per_design,
            "designs": rows,
        }, fh, indent=2)
    print(f"  saved -> {args.out}")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
