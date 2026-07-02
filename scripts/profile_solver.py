"""A2: profile the per-wavelength solver loop to attribute time.

Confirms/refutes the claim (README/EXPERIMENTS.md) that `set_freq_k` redundantly
re-expands the plane-wave/geometry basis every wavelength, which would make a
geometry-only caching adapter a real speedup (separate from the GPU-offload
adapter in `rcwa4d/backend.py`).

Usage:
    python -m scripts.profile_solver --N 2 --n-lam 6
    python -m scripts.profile_solver --N 3 --n-lam 3
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats

import numpy as np

from data_generation.parameter_sampler import DesignParams
from data_generation.simulate_spectra import simulate_design


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--n-lam", type=int, default=6)
    p.add_argument("--slab", default="Si3N4")
    args = p.parse_args()

    design = DesignParams(theta_deg=15.0, thickness=0.30, gap=0.10, radius=0.25)
    wl = np.linspace(600, 850, args.n_lam)

    profiler = cProfile.Profile()
    profiler.enable()
    simulate_design(design, wl, N=args.N, slab_material=args.slab)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(25)
    print(stream.getvalue())

    # Also break down by function name (not call path) to see hot leaves.
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2).sort_stats("tottime")
    stats2.print_stats(20)
    print(stream2.getvalue())


if __name__ == "__main__":
    main()
