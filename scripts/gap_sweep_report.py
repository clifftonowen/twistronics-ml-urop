"""Per-base report for gap sweeps (fixed theta/t/r, gap stepped).

Plots peak|CD|, fitted Q and f_peak against d/lambda (= gap * f_peak, the gap
layer being vacuum) for each sweep shard. A jump in f_peak between neighbouring
gaps larger than the window half-width means a different resonance has become
the strongest peak, so the curve should not be read as smooth across it.

Usage:
    python -m scripts.gap_sweep_report --shards peak_gapsweep_b1 [peak_gapsweep_b2 ...]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from scripts.compare_gap_qw import ROOT, fitted_q, load


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--half-width", type=float, default=0.03)
    p.add_argument("--fig", default=os.path.join(ROOT, "docs", "stage_c", "figures", "gap_sweep.png"))
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name in args.shards:
        D = load(name)
        D["q"] = fitted_q(D)
        order = np.argsort(D["gap"])
        gap, fp, cd, q = D["gap"][order], D["f_peak"][order], D["peak_cd"][order], D["q"][order]
        dl = gap * fp
        jumps = np.abs(np.diff(fp)) / fp[:-1] > args.half_width

        print(f"\n=== {name}: {gap.size} gaps ===")
        print("   gap   d/lam  f_peak  peak|CD|    Q   switch")
        for i in range(gap.size):
            sw = "  <- mode switch" if i > 0 and jumps[i - 1] else ""
            qs = f"{q[i]:5.0f}" if np.isfinite(q[i]) else "    -"
            print(f"  {gap[i]:.3f}  {dl[i]:.3f}  {fp[i]:.3f}   {cd[i]:.3f}   {qs}{sw}")
        print(f"  peak|CD| span {cd.max() - cd.min():.3f} (max at d/lam {dl[np.argmax(cd)]:.3f}); "
              f"{int(jumps.sum())} mode switch(es)")
        if np.isfinite(q).sum() >= 2:
            print(f"  Q range {np.nanmin(q):.0f}-{np.nanmax(q):.0f} "
                  f"(ratio {np.nanmax(q) / np.nanmin(q):.1f}x)")

        for a, y in zip(ax, (cd, q, fp)):
            line, = a.plot(dl, y, "o-", ms=4, label=name)
            for i in np.flatnonzero(jumps):
                a.axvline(0.5 * (dl[i] + dl[i + 1]), color=line.get_color(), ls=":", lw=0.8)

    for a, lab in zip(ax, ("peak |CD|", "fitted Q", "f_peak (a/lambda)")):
        a.axvline(0.25, ls="--", color="gray", lw=1)
        a.set_xlabel("d / lambda at CD peak")
        a.set_ylabel(lab)
    ax[0].legend(fontsize=8)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=150)
    print(f"\nfigure -> {args.fig}")


if __name__ == "__main__":
    main()
