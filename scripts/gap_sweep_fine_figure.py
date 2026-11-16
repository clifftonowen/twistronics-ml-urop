"""Summary figure for the gap sweeps, using only the +-1% window measurements.

Three panels: strongest |CD| against d/lambda for each tracked resonance, the
+/- swing of the two close pairs, and example CD spectra at each pair's largest
swing. d/lambda is gap * (frequency of the strongest CD point in the window),
the gap layer being vacuum.

B1's swing is drawn only from gap 0.282 up. Below that its two lobes are further
apart than a +-1% window spans, so the swing would not be comparable.

Usage:
    python -m scripts.gap_sweep_fine_figure
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "datasets", "raw")
GAP_COL = 2

SERIES = {
    "B1 pair (f≈0.71)": ("peak_gapsweep_b1_fine", "peak_gapsweep_b1_fine2", "peak_gapsweep_b1_fine3"),
    "B2 pair (f≈0.71)": ("peak_gapsweep_b2_pair", "peak_gapsweep_b2_pair2"),
    "B2 single mode (f≈0.66)": ("peak_gapsweep_b2_fine",),
}
COLORS = {
    "B1 pair (f≈0.71)": "#2a78d6",
    "B2 pair (f≈0.71)": "#eb6834",
    "B2 single mode (f≈0.66)": "#1baf7a",
}
MARKERS = {"B1 pair (f≈0.71)": "o", "B2 pair (f≈0.71)": "s", "B2 single mode (f≈0.66)": "^"}
MIN_B1_SWING_GAP = 0.28
INK, MUTED, GRID = "#222222", "#666666", "#E6E6E6"


def load_series(shards: tuple[str, ...]) -> list[dict]:
    rows = []
    for shard in shards:
        d = np.load(os.path.join(RAW, f"{shard}.npz"), allow_pickle=True)
        for i in range(d["X"].shape[0]):
            f, cd = d["window_freqs"][i], d["window_cd"][i]
            j = int(np.argmax(np.abs(cd)))
            gap = float(d["X"][i, GAP_COL])
            rows.append({
                "gap": gap, "f": f, "cd": cd,
                "d_lam": gap * float(f[j]),
                "peak": float(abs(cd[j])),
                "swing": float(cd.max() - cd.min()),
            })
    return sorted(rows, key=lambda r: r["gap"])


def style(ax, ylabel: str, xlabel: str = "d / λ at the CD peak") -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fig", default=os.path.join(ROOT, "docs", "stage_c", "figures", "gap_sweep_fine.png"))
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {name: load_series(shards) for name, shards in SERIES.items()}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

    for name, rows in data.items():
        kw = dict(color=COLORS[name], marker=MARKERS[name], ms=6, lw=2, mec="white", mew=1, label=name)
        ax[0].plot([r["d_lam"] for r in rows], [r["peak"] for r in rows], **kw)
        if "pair" in name:
            keep = [r for r in rows if not name.startswith("B1") or r["gap"] >= MIN_B1_SWING_GAP]
            ax[1].plot([r["d_lam"] for r in keep], [r["swing"] for r in keep], **kw)

    for a, ylabel in zip(ax[:2], ("strongest |CD|", "CD swing (+ peak minus - peak)")):
        a.axvline(0.25, ls="--", color=MUTED, lw=1)
        style(a, ylabel)
    ax[0].set_ylim(0, 0.6)
    ax[1].set_ylim(0, 1.05)
    ax[0].text(0.253, 0.03, "λ/4", color=MUTED)
    ax[1].text(0.253, 0.05, "λ/4", color=MUTED)
    ax[0].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    ax[1].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.1))

    for name, gap_target in (("B1 pair (f≈0.71)", 0.282), ("B2 pair (f≈0.71)", 0.310)):
        r = min(data[name], key=lambda x: abs(x["gap"] - gap_target))
        centre = float(r["f"][np.argmax(r["cd"])] + r["f"][np.argmin(r["cd"])]) / 2
        wavelength_shift = 500.0 / r["f"] - 500.0 / centre
        ax[2].plot(wavelength_shift, r["cd"], color=COLORS[name], lw=2,
                   label=f"{name.split(' pair')[0]}, gap {r['gap']:.3f}")
    ax[2].axhline(0, color=MUTED, lw=0.8)
    ax[2].set_xlim(-8, 8)
    ax[2].set_ylim(-0.6, 0.6)
    ax[2].legend(frameon=False, fontsize=9, loc="upper right")
    style(ax[2], "CD", "wavelength from the pair's midpoint (nm, a = 500 nm)")

    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=160, facecolor="white")
    print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
