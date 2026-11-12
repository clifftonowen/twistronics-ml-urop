"""Heatmaps of strongest |CD| and CD swing over a thickness x gap grid around one base.

Merges the grid shard with existing gap-sweep shards at the base thickness, so the
base row is reused rather than re-run. Cells whose window centre jumps by more
than --switch-frac from every neighbour are outlined: a different resonance has
probably become the strongest there, and the cell should not be read as part of
the same surface.

Usage:
    python -m scripts.grid_report
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "datasets", "raw")
DEFAULT_SHARDS = ("peak_grid_b1", "peak_gapsweep_b1_fine", "peak_gapsweep_b1_fine2",
                  "peak_gapsweep_b1_fine3")
INK, MUTED = "#222222", "#666666"


def load_cells(shards, gaps, thicknesses) -> dict:
    cells = {}
    for shard in shards:
        d = np.load(os.path.join(RAW, f"{shard}.npz"), allow_pickle=True)
        for i in range(d["X"].shape[0]):
            t, g = round(float(d["X"][i, 1]), 3), round(float(d["X"][i, 2]), 3)
            if t not in thicknesses or g not in gaps:
                continue
            cd = d["window_cd"][i]
            cells[(t, g)] = {
                "peak": float(np.abs(cd).max()),
                "swing": float(cd.max() - cd.min()),
                "f": float(d["f_peak"][i]),
            }
    return cells


def mode_switches(cells, thicknesses, gaps, frac) -> set:
    flagged = set()
    for a, t in enumerate(thicknesses):
        for b, g in enumerate(gaps):
            if (t, g) not in cells:
                continue
            nbrs = [cells[(thicknesses[a + da], gaps[b + db])]["f"]
                    for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= a + da < len(thicknesses) and 0 <= b + db < len(gaps)
                    and (thicknesses[a + da], gaps[b + db]) in cells]
            f = cells[(t, g)]["f"]
            if nbrs and all(abs(f / n - 1) > frac for n in nbrs):
                flagged.add((t, g))
    return flagged


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", default=list(DEFAULT_SHARDS))
    p.add_argument("--thicknesses", type=float, nargs="+", default=[0.36, 0.38, 0.40, 0.42, 0.44])
    p.add_argument("--gaps", type=float, nargs="+", default=[0.226, 0.254, 0.282, 0.310])
    p.add_argument("--switch-frac", type=float, default=0.02)
    p.add_argument("--fig", default=os.path.join(ROOT, "docs", "stage_c", "figures", "grid_b1.png"))
    args = p.parse_args(argv)

    ts, gs = [round(t, 3) for t in args.thicknesses], [round(g, 3) for g in args.gaps]
    cells = load_cells(args.shards, gs, ts)
    flagged = mode_switches(cells, ts, gs, args.switch_frac)

    print("t     gap    |CD|   swing  f_peak  switch")
    for t in ts:
        for g in gs:
            c = cells.get((t, g))
            if c:
                print(f"{t:.2f}  {g:.3f}  {c['peak']:.3f}  {c['swing']:.2f}   {c['f']:.4f}  "
                      f"{'yes' if (t, g) in flagged else ''}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, key, title in ((axes[0], "peak", "strongest |CD|"), (axes[1], "swing", "CD swing (+ peak minus - peak)")):
        M = np.full((len(ts), len(gs)), np.nan)
        for a, t in enumerate(ts):
            for b, g in enumerate(gs):
                if (t, g) in cells:
                    M[a, b] = cells[(t, g)][key]
        im = ax.imshow(M, cmap="Blues", origin="lower", aspect="auto")
        for a, t in enumerate(ts):
            for b, g in enumerate(gs):
                if np.isfinite(M[a, b]):
                    v = M[a, b]
                    dark = v > np.nanmin(M) + 0.6 * (np.nanmax(M) - np.nanmin(M))
                    ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=10,
                            color="white" if dark else INK)
                if (t, g) in flagged:
                    ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2, ls="--"))
        ax.set_xticks(range(len(gs)), [f"{g:.3f}" for g in gs])
        ax.set_yticks(range(len(ts)), [f"{t:.2f}" for t in ts])
        ax.set_xlabel("gap (a)", color=INK)
        ax.set_ylabel("slab thickness (a)", color=INK)
        ax.set_title(title, color=INK, fontsize=11)
        ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.text(0.5, -0.03, "dashed outline: window centre jumps >2% from all neighbours (likely a different resonance)",
             color=MUTED, fontsize=8, ha="center")
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=160, facecolor="white", bbox_inches="tight")
    print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
