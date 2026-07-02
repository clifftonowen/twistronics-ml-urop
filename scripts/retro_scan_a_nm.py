"""B3: demonstrate/measure the free post-hoc a_nm rescan on an existing shard.

Because v1 materials are non-dispersive, solver output depends only on the
normalized frequency f = a/lambda (and angle) -- not on a_nm separately. So for
any shard simulated on a WIDE f grid, re-picking a_nm after the fact slides a
[lam_lo, lam_hi] window across that f grid at zero solver cost. This script
scans a_nm, keeps only the in-window points, and reports the best achievable
peak |CD| per design and in aggregate.

This only pays off if the shard's simulated f range is WIDER than the target
window's ratio (lam_max/lam_min = 850/600 ~ 1.417); otherwise every other a_nm
strictly loses coverage relative to whatever a_nm the shard was already tuned
to (see EXPERIMENTS.md finding on v0_n120, where the two ratios are equal by
construction and the scan is a no-op).

Usage:
    python -m scripts.retro_scan_a_nm --shard wide_f_smoke
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True)
    p.add_argument("--lam-min", type=float, default=600.0)
    p.add_argument("--lam-max", type=float, default=850.0)
    p.add_argument("--a-nm-lo", type=float, default=300.0)
    p.add_argument("--a-nm-hi", type=float, default=900.0)
    p.add_argument("--n-a-nm", type=int, default=61)
    args = p.parse_args()

    d = np.load(os.path.join(RAW, f"{args.shard}.npz"))
    freqs = d["freqs"]
    CD = d["CD"]
    n_designs = CD.shape[0]

    f_lo, f_hi = freqs.min(), freqs.max()
    print(f"shard={args.shard}  n_designs={n_designs}  simulated f range=[{f_lo:.4f},{f_hi:.4f}] "
          f"(ratio {f_hi/f_lo:.3f})  target window ratio={args.lam_max/args.lam_min:.3f}")

    a_nm_grid = np.linspace(args.a_nm_lo, args.a_nm_hi, args.n_a_nm)
    best_peak = np.full(n_designs, -np.inf)
    best_a_nm = np.full(n_designs, np.nan)
    baseline_a_nm = 500.0  # the a_nm every prior shard was generated at

    for a_nm in a_nm_grid:
        wl = a_nm / freqs
        mask = (wl >= args.lam_min) & (wl <= args.lam_max)
        if mask.sum() == 0:
            continue
        peak = np.max(np.abs(CD[:, mask]), axis=1)
        improve = peak > best_peak
        best_peak[improve] = peak[improve]
        best_a_nm[improve] = a_nm

    # baseline: fixed a_nm=500 (or shard's own a_nm), same in-window restriction
    wl_base = baseline_a_nm / freqs
    mask_base = (wl_base >= args.lam_min) & (wl_base <= args.lam_max)
    if mask_base.sum() > 0:
        baseline_peak = np.max(np.abs(CD[:, mask_base]), axis=1)
    else:
        baseline_peak = np.zeros(n_designs)

    gain = best_peak - baseline_peak
    print(f"baseline (a_nm={baseline_a_nm:.0f}) median peak|CD| in-window: {np.median(baseline_peak):.4f}")
    print(f"best-over-a_nm scan   median peak|CD| in-window: {np.median(best_peak):.4f}")
    print(f"median gain from scanning a_nm: {np.median(gain):.4f}  "
          f"(designs improved: {(gain > 1e-6).sum()}/{n_designs})")


if __name__ == "__main__":
    main()
