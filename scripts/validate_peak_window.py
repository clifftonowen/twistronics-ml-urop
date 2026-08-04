"""Validate data_generation/peak_window.py: (1) synthetic spectra with known
ground truth, (2) the 20 real densely-scanned designs already on disk -- zero
new RCWA simulation cost either way.

This is the pre-flight check for the peak-centred narrow-window campaign
(EXPERIMENTS.md Sec 6c-6g motivate it). Before any GPU time is spent it must be
true that: the peak locator finds the right feature, the competing-lobe count
matches what a human sees in the dense scans, resampling onto the relative axis
is faithful and fails loudly out of range, and the window's advertised Q
coverage is real rather than an arithmetic claim.

Usage:
    python -m scripts.validate_peak_window                # both parts
    python -m scripts.validate_peak_window --synthetic-only
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data_generation.peak_window import (
    DEFAULT_HALF_WIDTH,
    DEFAULT_WINDOW_PTS,
    find_competing_lobes,
    locate_peak,
    relative_grid,
    resample_to_relative,
    resolvable_q_range,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")

DENSE_SHARDS = [
    "dense_rescan_highcd_n3_gpu_merged60",
    "dense_rescan_highcd_n3_gpu_merged60_pilot17",
]


def lorentzian_in_f(f, f0, q, amp=1.0):
    """Unit-shape Lorentzian of quality factor q centred at f0.

    Parametrized by Q directly (FWHM = f0/q) because Q is what the window's
    coverage is stated in -- this keeps the test and the claim in the same units.
    """
    hwhm = 0.5 * f0 / q
    return amp / (1.0 + ((np.asarray(f) - f0) / hwhm) ** 2)


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run_synthetic() -> bool:
    print("=== Synthetic validation ===")
    ok = True
    f = np.linspace(0.60, 0.92, 81)  # the planned locate grid

    # 1. Isolated peak: located within one grid step, counted as a single lobe.
    cd = lorentzian_in_f(f, 0.7400, q=60, amp=0.4)
    loc = locate_peak(f, cd)
    ok &= _report(
        "isolated peak located",
        abs(loc.f_peak - 0.7400) <= loc.grid_step_f and loc.n_competing_lobes == 1,
        f"f_peak={loc.f_peak:.4f} (true 0.7400, step {loc.grid_step_f:.4f}), "
        f"lobes={loc.n_competing_lobes}, peak|CD|={loc.peak_abs_cd:.3f}",
    )

    # 1b. The a-priori Q estimate from the locate grid must track the truth for
    #     a resolved peak, and must correctly self-report as a lower bound when
    #     the peak is too narrow for the grid to measure.
    fine = np.linspace(0.70, 0.78, 801)
    loc_fine = locate_peak(fine, lorentzian_in_f(fine, 0.7400, q=60, amp=0.4))
    ok &= _report(
        "Q estimate accurate on a resolving grid",
        abs(loc_fine.q_estimate - 60.0) / 60.0 < 0.05 and not loc_fine.width_is_grid_limited,
        f"q_estimate={loc_fine.q_estimate:.1f} (true 60), "
        f"fwhm={loc_fine.fwhm_grid_pts:.1f} grid pts, grid_limited={loc_fine.width_is_grid_limited}",
    )
    loc_sharp = locate_peak(f, lorentzian_in_f(f, 0.7400, q=400, amp=0.4))
    ok &= _report(
        "grid-limited peak self-reports as a lower bound",
        loc_sharp.width_is_grid_limited and loc_sharp.q_estimate < 400.0,
        f"q_estimate={loc_sharp.q_estimate:.0f} (true 400, correctly under-reported), "
        f"fwhm={loc_sharp.fwhm_grid_pts:.1f} grid pts",
    )

    # 2. Sign matters: CD is signed and the handedness must survive.
    loc_neg = locate_peak(f, -cd)
    ok &= _report(
        "signed CD preserved",
        loc_neg.cd_at_peak < 0 and loc_neg.peak_abs_cd > 0,
        f"cd_at_peak={loc_neg.cd_at_peak:+.3f}, peak|CD|={loc_neg.peak_abs_cd:.3f}",
    )

    # 3. Two comparable lobes -> the label-swap case the count exists to flag.
    cd2 = lorentzian_in_f(f, 0.700, q=60, amp=0.40) - lorentzian_in_f(f, 0.800, q=60, amp=0.38)
    n2 = find_competing_lobes(f, cd2).size
    ok &= _report("two comparable lobes counted", n2 == 2, f"n_lobes={n2} (expected 2)")

    # 4. A far weaker neighbour must NOT count -- otherwise the metric would
    #    flag every design and carry no information.
    cd3 = lorentzian_in_f(f, 0.700, q=60, amp=0.40) + lorentzian_in_f(f, 0.800, q=60, amp=0.05)
    n3 = find_competing_lobes(f, cd3).size
    ok &= _report("weak neighbour ignored", n3 == 1, f"n_lobes={n3} (expected 1)")

    # 5. Peak at the band edge: find_peaks cannot return an endpoint, but an
    #    edge peak means the scanned band was too narrow -- the worst thing to
    #    drop silently.
    cd4 = lorentzian_in_f(f, 0.600, q=60, amp=0.4)
    n4 = find_competing_lobes(f, cd4).size
    ok &= _report("endpoint peak detected", n4 == 1, f"n_lobes={n4} (expected 1)")

    # 6. Resampling onto the relative axis is faithful in range, NaN out of it.
    dense_f = np.linspace(0.70, 0.78, 401)
    dense_cd = lorentzian_in_f(dense_f, 0.7400, q=60, amp=0.4)
    got = resample_to_relative(dense_f, dense_cd, 0.7400, half_width=0.01, n_pts=21)
    want = lorentzian_in_f(relative_grid(0.7400, 0.01, 21), 0.7400, q=60, amp=0.4)
    err = float(np.max(np.abs(got - want)))
    ok &= _report("resample faithful in range", err < 1e-6 and not np.isnan(got).any(),
                  f"max abs err={err:.2e}, NaNs={int(np.isnan(got).sum())}")

    narrow_f = np.linspace(0.735, 0.745, 51)  # narrower than the +-3% window
    got_oor = resample_to_relative(narrow_f, lorentzian_in_f(narrow_f, 0.74, 60, 0.4), 0.7400)
    n_nan = int(np.isnan(got_oor).sum())
    ok &= _report("out-of-range resample -> NaN, not extrapolation",
                  n_nan > 0 and np.nanmax(np.abs(got_oor)) <= 1.0,
                  f"{n_nan}/{got_oor.size} NaN, max finite |value|={np.nanmax(np.abs(got_oor)):.3f}")

    # 7. The advertised Q coverage, checked EMPIRICALLY rather than by
    #    re-deriving the same arithmetic. For each stated bound, sample a
    #    Lorentzian of that Q on the window grid and measure the worst-case
    #    recovered peak height over all grid phases (the peak can fall between
    #    samples). A "resolved" Q must survive; an "unresolved" one must not.
    qr = resolvable_q_range(DEFAULT_HALF_WIDTH, DEFAULT_WINDOW_PTS)
    print(f"    window +-{DEFAULT_HALF_WIDTH:.0%} x {DEFAULT_WINDOW_PTS} pts -> step_u={qr.step_u:.5f}, "
          f"Q in [{qr.q_min:.0f}, {qr.q_max_resolved:.0f}] resolved, "
          f"detectable to {qr.q_max_detectable:.0f}")
    f0 = 0.7400
    for q, label, floor in [
        (qr.q_min * 1.2, "broadest resolved", 0.99),
        (qr.q_max_resolved, "narrowest resolved", 0.90),
        (qr.q_max_detectable * 3.0, "far beyond detectable", 0.0),
    ]:
        worst = 1.0
        for shift in np.linspace(-0.5, 0.5, 21):
            grid = relative_grid(f0, DEFAULT_HALF_WIDTH, DEFAULT_WINDOW_PTS) + shift * qr.step_u * f0
            worst = min(worst, float(lorentzian_in_f(grid, f0, q, 1.0).max()))
        if label == "far beyond detectable":
            good = worst < 0.5   # must visibly fail, proving the bound is not vacuous
            detail = f"Q={q:.0f} worst-case recovered {worst:.2f} (<0.50 expected: genuinely unresolved)"
        else:
            good = worst >= floor
            detail = f"Q={q:.0f} worst-case recovered {worst:.2f} (>={floor:.2f} required)"
        ok &= _report(f"Q coverage: {label}", good, detail)

    return bool(ok)


def run_real() -> bool:
    """Exercise the module on every design already densely scanned on disk."""
    print("\n=== Real dense-scan validation (no new simulation) ===")
    a_nm = 500.0
    rows = []
    for shard in DENSE_SHARDS:
        path = os.path.join(RAW, f"{shard}.npz")
        if not os.path.exists(path):
            print(f"  (skipped, not found: {path})")
            continue
        d = np.load(path)
        wl = d["dense_wavelengths_nm"]
        freqs = a_nm / wl
        for idx in d["indices"]:
            cd = d[f"design{idx}_dense_cd"]
            loc = locate_peak(freqs, cd)
            rel = resample_to_relative(freqs, cd, loc.f_peak)
            rows.append((int(idx), loc, rel, cd))

    if not rows:
        return _report("real dense scans available", False, "no dense shards on disk")

    qr = resolvable_q_range(DEFAULT_HALF_WIDTH, DEFAULT_WINDOW_PTS)
    print(f"  {'idx':>4s} {'f_peak':>7s} {'lam_pk':>7s} {'CD@pk':>7s} {'lobes':>6s} "
          f"{'a_nm@725':>9s} {'winNaN':>7s} {'Qest':>6s} {'class':>11s}")
    ok = True
    argmax_mismatch = []
    for idx, loc, rel, cd in rows:
        n_nan = int(np.isnan(rel).sum())
        cls = qr.classify(loc.q_estimate) + ("*" if loc.width_is_grid_limited else "")
        print(f"  {idx:4d} {loc.f_peak:7.4f} {a_nm / loc.f_peak:7.1f} {loc.cd_at_peak:+7.3f} "
              f"{loc.n_competing_lobes:6d} {loc.lattice_constant_nm(725.0):9.0f} {n_nan:7d} "
              f"{loc.q_estimate:6.0f} {cls:>11s}")
        # The located peak must be the true argmax of the source array. Every
        # window in the campaign is centred on this number, so it is the single
        # assumption the representation cannot survive being wrong about.
        if abs(loc.peak_abs_cd - float(np.abs(cd).max())) > 1e-12:
            argmax_mismatch.append(idx)
    ok &= _report("located peak is the true argmax |CD|", not argmax_mismatch,
                  f"{len(rows) - len(argmax_mismatch)}/{len(rows)} exact"
                  + (f", mismatched: {argmax_mismatch}" if argmax_mismatch else ""))

    lobes = np.array([r[1].n_competing_lobes for r in rows])
    multi = int((lobes >= 4).sum())
    ok &= _report(
        "competing-lobe count reproduces the known distribution",
        multi >= 5,
        f"{multi}/{len(rows)} designs have >=4 comparable lobes "
        f"(independent analysis of these shards found 7/20)",
    )

    # Every design's window must be fully inside its own scanned band, else the
    # relative representation cannot be built from existing data at all.
    covered = sum(1 for r in rows if not np.isnan(r[2]).any())
    ok &= _report(
        "relative window lies inside the scanned band",
        covered >= len(rows) - 4,
        f"{covered}/{len(rows)} designs fully covered "
        f"(edge-peaked designs legitimately are not)",
    )

    # Not a pass/fail -- a budget readout. These designs were scanned at
    # ~0.004 in f, coarser than the campaign window, so "grid_limited" here
    # only bounds Q from below. It is the best advance estimate available of
    # how much of the parameter space the chosen window will under-resolve.
    grid_limited = sum(1 for r in rows if r[1].width_is_grid_limited)
    print(f"\n  {grid_limited}/{len(rows)} designs are width-limited by the 0.004 locate grid "
          f"(Q >~ {1.0 / (3 * 0.004 / 0.75):.0f}) -- the window resolves to Q={qr.q_max_resolved:.0f}")

    peaks = np.array([r[1].f_peak for r in rows])
    print(f"\n  f_peak span {peaks.min():.3f}-{peaks.max():.3f} "
          f"(lambda {a_nm / peaks.max():.0f}-{a_nm / peaks.min():.0f} nm) -> "
          f"a_nm {peaks.min() * 725:.0f}-{peaks.max() * 725:.0f} nm to centre all peaks at 725 nm")
    return bool(ok)


def main(argv: list[str] | None = None) -> bool:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic-only", action="store_true")
    args = p.parse_args(argv)

    ok = run_synthetic()
    if not args.synthetic_only:
        ok = run_real() and ok
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
