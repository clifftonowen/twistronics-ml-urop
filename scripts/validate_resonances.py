"""Validate data_generation/resonances.py: (1) synthetic Fano spectra with
known ground truth, (2) the 3 real densely-sampled designs already produced by
Step 3's aliasing diagnostic -- zero new RCWA simulation cost either way.

This is the go/no-go check for Step 4a (EXPERIMENTS.md Sec 6c / PLANS.md Sec
5.3): does fitting T_RCP/T_LCP as Fano resonances actually recover sensible,
stable (lambda0, Q, amplitude), well enough to justify investing in a
densified dataset for surrogate retraining next?

Usage:
    python -m scripts.validate_resonances                # both parts
    python -m scripts.validate_resonances --synthetic-only
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data_generation.resonances import (
    fano_lineshape,
    fit_spectrum,
    match_rcp_lcp_modes,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "datasets", "raw")
ARTIFACTS = os.path.join(ROOT, "models", "artifacts")


def synthetic_case(lam0s, gammas, qs, amps, bg=0.5, lam_range=(600, 850), n_lam=301, noise=0.0, seed=0):
    """Build a synthetic multi-Fano spectrum, rescaled to respect T in [0,1]
    (real T_RCP/T_LCP are transmission coefficients and can never leave that
    range -- `fit_resonance` now enforces this on FITTED curves, so the
    synthetic ground truth must respect it too or the comparison is invalid;
    an unscaled Fano with amp=0.3, q=2, bg=0.5 peaks at 1.7, already outside
    [0,1] by construction). The rescale is linear about bg (y -> bg + k*(y-bg))
    so it changes only overall amplitude, not lineshape/Q -- ground-truth Q
    is unaffected.
    """
    lam = np.linspace(*lam_range, n_lam)
    y = np.full_like(lam, bg)
    for l0, g, q, a in zip(lam0s, gammas, qs, amps):
        y += fano_lineshape(lam, l0, g, q, a, 0.0)

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    headroom_lo, headroom_hi = bg - 0.05, 0.95 - bg
    k_lo = headroom_lo / (bg - y_lo) if y_lo < bg else float("inf")
    k_hi = headroom_hi / (y_hi - bg) if y_hi > bg else float("inf")
    k = min(1.0, k_lo, k_hi)
    y = bg + k * (y - bg)

    if noise > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0, noise, size=y.shape)
    return lam, y


def run_synthetic() -> bool:
    """Return True if every recovered Q is within ~10% of ground truth."""
    print("=== Synthetic Fano validation ===")
    cases = [
        {"name": "single peak", "lam0s": [700.0], "gammas": [8.0], "qs": [2.0], "amps": [0.3]},
        {"name": "single dip", "lam0s": [720.0], "gammas": [5.0], "qs": [0.05], "amps": [-0.4]},
        {"name": "two well-separated", "lam0s": [650.0, 780.0], "gammas": [6.0, 10.0], "qs": [1.5, 0.2], "amps": [0.25, -0.2]},
        {"name": "two close (mode-pair-like)", "lam0s": [700.0, 706.0], "gammas": [4.0, 4.0], "qs": [1.0, -1.0], "amps": [0.2, -0.2]},
        {"name": "noisy single peak", "lam0s": [710.0], "gammas": [7.0], "qs": [1.0], "amps": [0.3], "noise": 0.005},
    ]
    all_ok = True
    for case in cases:
        lam, y = synthetic_case(
            case["lam0s"], case["gammas"], case["qs"], case["amps"],
            noise=case.get("noise", 0.0),
        )
        fits = fit_spectrum(lam, y, window_nm=60.0)
        good_fits = sorted([f for f in fits if f.success and f.r2 > 0.8], key=lambda f: f.lam0)
        print(f"\n{case['name']}: {len(case['lam0s'])} injected, {len(good_fits)} recovered (r2>0.8)")
        # Match each TRUE resonance to its nearest recovered fit by lam0
        # (not positional order) -- avoids misattributing errors when a
        # spurious/unrelated fit happens to sort between two true values.
        used = set()
        for l0_true, g_true in sorted(zip(case["lam0s"], case["gammas"]), key=lambda t: t[0]):
            candidates_left = [(i, f) for i, f in enumerate(good_fits) if i not in used]
            if not candidates_left:
                print(f"  true lam0={l0_true:.1f}: NO recovered fit left to match  FAIL")
                all_ok = False
                continue
            i, f = min(candidates_left, key=lambda t: abs(t[1].lam0 - l0_true))
            used.add(i)
            Q_true = l0_true / g_true
            rel_err = abs(f.Q - Q_true) / Q_true
            ok = rel_err <= 0.10 and abs(f.lam0 - l0_true) <= g_true
            all_ok &= ok
            print(f"  true lam0={l0_true:.1f} Q={Q_true:.2f}  |  nearest fit lam0={f.lam0:.2f} Q={f.Q:.2f} "
                  f"r2={f.r2:.3f}  rel_err={rel_err:.1%}  {'OK' if ok else 'FAIL'}")
        extra = len(good_fits) - len(used)
        if extra > 0:
            print(f"  NOTE: {extra} extra recovered fit(s) beyond the {len(case['lam0s'])} injected "
                  f"(informational only -- not counted against the pass/fail gate)")
    print(f"\nSynthetic validation: {'PASS' if all_ok else 'FAIL'} (all recovered Q within 10%: {all_ok})")
    return all_ok


def run_real(shard: str = "highcd_n3_gpu_merged60") -> None:
    print("\n=== Real dense-scan validation (zero new GPU cost) ===")
    path = os.path.join(RAW, f"dense_rescan_{shard}.npz")
    d = np.load(path)
    indices = [int(i) for i in d["indices"]]
    dense_wl = d["dense_wavelengths_nm"]

    for idx in indices:
        t_rcp = d[f"design{idx}_dense_T_RCP"]
        t_lcp = d[f"design{idx}_dense_T_LCP"]
        cd = d[f"design{idx}_dense_cd"]
        peak_abs_cd = float(np.abs(cd).max())

        res_rcp = fit_spectrum(dense_wl, t_rcp, window_nm=60.0)
        res_lcp = fit_spectrum(dense_wl, t_lcp, window_nm=60.0)
        good_rcp = [f for f in res_rcp if f.success and f.r2 > 0.8]
        good_lcp = [f for f in res_lcp if f.success and f.r2 > 0.8]
        pairs = match_rcp_lcp_modes(res_rcp, res_lcp, max_split_nm=15.0, min_r2=0.9)

        print(f"\ndesign idx={idx}  (known dense peak|CD|={peak_abs_cd:.4f})")
        print(f"  RCP: {len(res_rcp)} candidates, {len(good_rcp)} good fits (r2>0.8)")
        print(f"  LCP: {len(res_lcp)} candidates, {len(good_lcp)} good fits (r2>0.8)")
        print(f"  matched mode pairs: {len(pairs)}  (sorted by predicted |CD|, strongest first)")
        for p in sorted(pairs, key=lambda p: -p.predicted_peak_abs_cd):
            # CD at the RCP mode's lam0, computed from the ACTUAL dense data
            # (not the fit) as an independent cross-check against the
            # predicted score the matching optimized for.
            i_nearest = int(np.argmin(np.abs(dense_wl - p.rcp.lam0)))
            cd_at_mode = float(cd[i_nearest])
            print(f"    RCP lam0={p.rcp.lam0:.1f} Q={p.rcp.Q:.1f}  |  "
                  f"LCP lam0={p.lcp.lam0:.1f} Q={p.lcp.Q:.1f}  |  "
                  f"split={p.splitting_nm:.2f}nm  |  predicted|CD|={p.predicted_peak_abs_cd:.4f}  |  "
                  f"CD@mode(actual)={cd_at_mode:+.4f}")

        _plot_design(shard, idx, dense_wl, t_rcp, t_lcp, res_rcp, res_lcp)


def _plot_design(shard, idx, lam, t_rcp, t_lcp, res_rcp, res_lcp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y, fits, label in zip(axes, [t_rcp, t_lcp], [res_rcp, res_lcp], ["T_RCP", "T_LCP"]):
        ax.plot(lam, y, "-", lw=1, color="0.3", label=f"{label} (dense)")
        dense_fine = np.linspace(lam.min(), lam.max(), 600)
        for f in fits:
            if f.success and f.r2 > 0.8:
                mask = np.abs(dense_fine - f.lam0) <= f.window_nm / 2
                ax.plot(dense_fine[mask], fano_lineshape(dense_fine[mask], f.lam0, f.gamma, f.q, f.amp, f.bg),
                        "--", lw=1.5, label=f"fit lam0={f.lam0:.0f} Q={f.Q:.0f}")
        ax.set(xlabel="wavelength (nm)", ylabel=label, title=f"design {idx}: {label}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(ARTIFACTS, exist_ok=True)
    p = os.path.join(ARTIFACTS, f"resonance_fit_{shard}_idx{idx}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  plot: {p}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--shard", default="highcd_n3_gpu_merged60")
    args = p.parse_args(argv)

    synthetic_ok = run_synthetic()
    if not args.synthetic_only:
        run_real(args.shard)
    print(f"\n{'='*60}\nSynthetic gate: {'PASS' if synthetic_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
