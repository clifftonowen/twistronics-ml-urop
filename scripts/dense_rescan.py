"""Aliasing diagnostic: does the 26-point/600-850nm campaign grid (~10nm
spacing) resolve the CD spectrum, or does it alias sharp, high-Q resonant
structure into something that looks like noise to a regressor?

For each requested design (by row index into an existing shard's raw X), this
re-simulates that exact structure at a much denser wavelength grid, then
compares the design's *existing* coarse (26-point) CD -- cubic-spline
interpolated onto the dense grid -- against the freshly-simulated dense CD.
A small RMS mismatch (relative to the dense CD's own std) means CD is smooth
at ~10nm scale and the campaign grid was fine (so a flat learning curve is a
data-volume problem, not a resolution problem). A large mismatch means the
coarse grid missed real structure -- direct evidence for redesigning the
target representation (denser grids and/or resonance fitting) before buying
more designs at the same 26-point grid.

Uses `simulate_design` directly (same solver call `generate_dataset.py` and
the campaign used); does NOT go through the sampler, since we want the exact
already-generated structures, not new LHS draws.

Usage:
    python -m scripts.dense_rescan --shard highcd_n3_gpu_merged60 \
        --indices 38 26 48 --n-lam 151 --device gpu
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.interpolate import CubicSpline

import data_generation  # noqa: F401 -- ensures rcwa4d is on sys.path
import rcwa4d.backend as backend
from data_generation.parameter_sampler import PARAM_NAMES, DesignParams
from data_generation.simulate_spectra import simulate_design

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")
ARTIFACTS = os.path.join(ROOT, "models", "artifacts")


def _checkpoint(out_path: str, results: dict) -> None:
    """Overwrite the output npz with whatever designs have completed so far.

    Mirrors the per-design checkpointing added to generate_dataset.py this
    session -- a multi-hour run must survive being killed partway through.
    """
    np.savez_compressed(out_path, **results)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", required=True, help="source shard providing X/CD/wavelengths")
    p.add_argument("--indices", type=int, nargs="+", required=True, help="row indices into the shard's X to rescan")
    p.add_argument("--n-lam", type=int, default=151, help="dense wavelength grid points")
    p.add_argument("--lam-min", type=float, default=600.0)
    p.add_argument("--lam-max", type=float, default=850.0)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--a-nm", type=float, default=500.0)
    p.add_argument("--slab", default="Si3N4")
    p.add_argument("--hole", default="air")
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--out", default=None, help="output npz path (default: datasets/raw/dense_rescan_<shard>.npz)")
    args = p.parse_args(argv)

    backend.set_device(args.device)

    raw = np.load(os.path.join(RAW, f"{args.shard}.npz"))
    X = raw["X"]
    coarse_wl = raw["wavelengths_nm"]
    coarse_cd = raw["CD"]

    dense_wl = np.linspace(args.lam_min, args.lam_max, args.n_lam)

    out_path = args.out or os.path.join(RAW, f"dense_rescan_{args.shard}.npz")
    results: dict = {
        "indices": np.array(args.indices),
        "dense_wavelengths_nm": dense_wl,
        "coarse_wavelengths_nm": coarse_wl,
    }
    rms_report = []

    for idx in args.indices:
        params = DesignParams(**dict(zip(PARAM_NAMES, X[idx])))
        print(f"design idx={idx} params={dict(zip(PARAM_NAMES, np.round(X[idx], 4)))}")
        t0 = time.perf_counter()
        res = simulate_design(
            params, dense_wl, a_nm=args.a_nm, N=args.N,
            slab_material=args.slab, hole_material=args.hole,
        )
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s ({elapsed / args.n_lam:.2f} s/wavelength)")

        # Interpolate the EXISTING coarse CD onto the dense grid and compare
        # against the freshly-simulated dense CD -- this isolates "did the
        # coarse grid miss real structure" from "is the design itself noisy".
        spline = CubicSpline(coarse_wl, coarse_cd[idx])
        coarse_interp_on_dense = spline(dense_wl)
        diff = res.cd - coarse_interp_on_dense
        rms_mismatch = float(np.sqrt(np.mean(diff ** 2)))
        dense_std = float(np.std(res.cd))
        ratio = rms_mismatch / dense_std if dense_std > 1e-12 else float("nan")
        rms_report.append({
            "index": int(idx), "rms_mismatch": rms_mismatch,
            "dense_cd_std": dense_std, "ratio": ratio,
            "dense_cd_peak_abs": float(np.abs(res.cd).max()),
            "energy_residual_max": float(res.energy_residual().max()),
        })
        print(f"  RMS(dense - coarse_interp) = {rms_mismatch:.4f}  |  dense CD std = {dense_std:.4f}  |  ratio = {ratio:.3f}")

        results[f"design{idx}_dense_cd"] = res.cd
        results[f"design{idx}_dense_T_RCP"] = res.T_RCP
        results[f"design{idx}_dense_T_LCP"] = res.T_LCP
        results[f"design{idx}_coarse_cd"] = coarse_cd[idx]
        results[f"design{idx}_coarse_interp_on_dense"] = coarse_interp_on_dense
        results[f"design{idx}_energy_residual"] = res.energy_residual()
        _checkpoint(out_path, results)  # survive an interruption mid-run

    print("\nSummary (ratio = RMS mismatch / dense CD std; large => aliasing):")
    for r in rms_report:
        print(f"  idx={r['index']:3d}  ratio={r['ratio']:.3f}  rms={r['rms_mismatch']:.4f}  "
              f"dense_std={r['dense_cd_std']:.4f}  peak|CD|={r['dense_cd_peak_abs']:.4f}")

    _try_plot(args.shard, args.indices, coarse_wl, coarse_cd, dense_wl, results, args.out or out_path)

    print(f"\nsaved -> {out_path}")
    return {"out_path": out_path, "rms_report": rms_report}


def _try_plot(shard, indices, coarse_wl, coarse_cd, dense_wl, results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), squeeze=False)
    for j, idx in enumerate(indices):
        ax = axes[0][j]
        ax.plot(dense_wl, results[f"design{idx}_dense_cd"], "-", lw=1, label="dense (new)")
        ax.plot(coarse_wl, coarse_cd[idx], "o", ms=4, label="coarse (campaign)")
        ax.set(xlabel="wavelength (nm)", ylabel="CD", title=f"design idx={idx}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Dense vs coarse CD rescan ({shard})")
    fig.tight_layout()
    plot_path = os.path.join(ARTIFACTS, f"dense_rescan_{shard}.png")
    os.makedirs(ARTIFACTS, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  plot: {plot_path}")


if __name__ == "__main__":
    main()
