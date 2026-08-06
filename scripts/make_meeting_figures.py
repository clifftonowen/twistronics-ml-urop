"""Regenerate the presentation figures for the 2026-07-23 supervisor meeting.

Every figure is rebuilt from the project's own artifacts (dataset shards, CV
reports, the convergence CSV, the dense-rescan npz) or drawn from scratch, in
one consistent style -- CPU only, no solver runs, no GPU cost.

    python scripts/make_meeting_figures.py            # all figures
    python scripts/make_meeting_figures.py --only 8 9 # just those

Figure numbers match slide order in docs/meeting_2026-07-23/slides.md.

Constants that cannot be recomputed from a stored artifact (wall-clock
benchmarks, the resonance-feature GP CV row) are collected in BENCHMARKS /
RESONANCE_CV below with a pointer to the log entry they come from, so a stale
number is easy to spot and fix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_generation.parameter_sampler import BOUNDS, HIGH_CD_BOUNDS  # noqa: E402
from data_generation.resonances import fit_spectrum  # noqa: E402

RAW = os.path.join(ROOT, "datasets", "raw")
PROC = os.path.join(ROOT, "datasets", "processed")
ARTIFACTS = os.path.join(ROOT, "models", "artifacts")
CONVERGENCE = os.path.join(ROOT, "datasets", "convergence")
OUT_DIR = os.path.join(ROOT, "docs", "meeting_2026-07-23", "figures")

# Okabe-Ito colourblind-safe palette.
BLUE, ORANGE, GREEN, VERMILLION = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, YELLOW, GREY = "#CC79A7", "#56B4E9", "#F0E442", "#666666"

# Wall-clock benchmarks (EXPERIMENTS.md Sec 6, "clean sole-job measurement" and
# the corrected cost model). These are measurements, not derivable from a file.
BENCHMARKS = {
    "n2_cpu_s_per_wl": 24.9 / 3,     # 3-wavelength A/B, 24.9 s total
    "n2_gpu_s_per_wl": 6.0 / 3,
    "n3_cpu_s_per_wl": 216.96,
    "n3_gpu_s_per_wl": 37.92,
    "n3_cpu_min_per_design": 94.0,   # 26-lambda design, authoritative figure
    "n3_gpu_min_per_design": 16.4,
}

# n=20 resonance-feature CV: the MLP row is in cv_report_resonance_pilot20.json;
# the GP row is reported in EXPERIMENTS.md Sec 6g but not stored as JSON.
RESONANCE_CV_GP = {"model_r2_mean": -0.505, "model_r2_std": 0.542}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    })


def save(fig, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    return path


def peak_abs_cd(cd: np.ndarray) -> np.ndarray:
    return np.max(np.abs(cd), axis=1)


def box(ax, x, y, w, h, text, face, edge, fontsize=11, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=face, edgecolor=edge, linewidth=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, linespacing=1.35)


def arrow(ax, xy_from, xy_to, color="#333333", style_="-|>", ls="-", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style_, mutation_scale=16,
                                 color=color, linestyle=ls, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


# --------------------------------------------------------------------------- #
# 01 -- pipeline schematic                                                     #
# --------------------------------------------------------------------------- #
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_xlim(0, 12), ax.set_ylim(0, 5), ax.axis("off"), ax.grid(False)

    stages = [
        (0.15, "Design params  X\n(θ, t, gap, r)", "sampled by Latin\nhypercube", GREEN, "done"),
        (2.55, "RCWA-4D solver\n($N_m$ = 3, GPU)", "5.7× faster\nafter gmatmul", GREEN, "done"),
        (4.95, "Dataset\nT_RCP, T_LCP → CD", "60 designs (N=3)\n+20 densified", GREEN, "done"),
        (7.35, "Forward surrogate\nX → spectrum", "T learnable ✓\nCD not yet ✗", ORANGE, "partial"),
        (9.75, "Inverse design\ntarget CD → X", "not started\n(gated on above)", GREY, "todo"),
    ]
    w, h, y = 2.1, 1.25, 2.6
    for x, label, sub, colour, _ in stages:
        box(ax, x, y, w, h, label, face=colour + "22", edge=colour, fontsize=11.5, weight="bold")
        ax.text(x + w / 2, y - 0.42, sub, ha="center", va="center", fontsize=10, color="#444444")
    for i in range(len(stages) - 1):
        arrow(ax, (stages[i][0] + w, y + h / 2), (stages[i + 1][0], y + h / 2))

    # Verification loop: every ML proposal goes back through the solver. Routed
    # around the outside of the row so it crosses neither boxes nor captions.
    yc, xr, xl = 1.25, 12.35, 2.42
    ax.plot([9.75 + w, xr], [y + h / 2, y + h / 2], color=VERMILLION, ls="--", lw=1.8)
    ax.plot([xr, xr], [y + h / 2, yc], color=VERMILLION, ls="--", lw=1.8)
    ax.plot([xr, xl], [yc, yc], color=VERMILLION, ls="--", lw=1.8)
    ax.plot([xl, xl], [yc, y + h / 2], color=VERMILLION, ls="--", lw=1.8)
    arrow(ax, (xl, y + h / 2), (2.55, y + h / 2), color=VERMILLION, ls="--")
    ax.text(7.35, 1.5, "every proposed design is re-simulated in RCWA-4D before it counts",
            ha="center", fontsize=11.5, color=VERMILLION, style="italic")
    ax.set_xlim(0, 12.8)

    ax.text(0.15, 4.45, "Phase 1: data generation", fontsize=12, color=GREEN, fontweight="bold")
    ax.text(7.35, 4.45, "Phase 2 / 3: learning", fontsize=12, color=ORANGE, fontweight="bold")
    ax.plot([0.1, 7.1], [4.25, 4.25], color=GREEN, lw=3, alpha=0.5)
    ax.plot([7.3, 11.9], [4.25, 4.25], color=ORANGE, lw=3, alpha=0.5)
    ax.set_title("Pipeline: where the project stands", pad=14)
    return save(fig, "fig01_pipeline.png")


# --------------------------------------------------------------------------- #
# 02 -- geometry / parametrization                                             #
# --------------------------------------------------------------------------- #
def fig02_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Top view: two square hole lattices, one rotated -> moire pattern.
    ax = axes[0]
    theta = np.deg2rad(15.0)
    n, r = 9, 0.16
    coords = np.arange(-n, n + 1, dtype=float)
    gx, gy = np.meshgrid(coords, coords)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rot = pts @ rot.T
    for p in pts:
        ax.add_patch(Circle(p, r, facecolor=BLUE, edgecolor="none", alpha=0.55))
    for p in pts_rot:
        ax.add_patch(Circle(p, r, facecolor=VERMILLION, edgecolor="none", alpha=0.55))
    lim = 5.2
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(3.6, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-", color=BLUE, lw=2))
    ax.annotate("", xy=(3.6 * np.cos(theta), 3.6 * np.sin(theta)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color=VERMILLION, lw=2))
    ax.text(3.0, 0.55, r"$\theta$", fontsize=17, color="#222222")
    ax.set_title("Top view: twist → moiré superlattice")
    ax.set_xlabel("two C₄ᵥ layers, twisted → C₄ : geometrically chiral", fontsize=11,
                  color="#444444", labelpad=8)

    # Side view: slab / gap / slab with the sampled lengths annotated.
    ax = axes[1]
    ax.set_xlim(0, 10), ax.set_ylim(0, 10), ax.axis("off"), ax.grid(False)
    slab_x, slab_w = 1.6, 6.4
    t_h, gap_h = 1.35, 1.1
    y_bot = 2.6
    ax.add_patch(Rectangle((slab_x, y_bot), slab_w, t_h, facecolor=VERMILLION,
                           alpha=0.35, edgecolor=VERMILLION, lw=2))
    ax.add_patch(Rectangle((slab_x, y_bot + t_h + gap_h), slab_w, t_h, facecolor=BLUE,
                           alpha=0.35, edgecolor=BLUE, lw=2))
    for cx in np.linspace(slab_x + 0.9, slab_x + slab_w - 0.9, 4):
        ax.add_patch(Rectangle((cx - 0.28, y_bot), 0.56, t_h, facecolor="white", edgecolor="none"))
        ax.add_patch(Rectangle((cx - 0.28 + 0.35, y_bot + t_h + gap_h), 0.56, t_h,
                               facecolor="white", edgecolor="none"))
    # dimension labels
    ax.annotate("", xy=(slab_x - 0.35, y_bot), xytext=(slab_x - 0.35, y_bot + t_h),
                arrowprops=dict(arrowstyle="<->", color="#222222"))
    ax.text(slab_x - 0.6, y_bot + t_h / 2, "t", fontsize=15, ha="right", va="center")
    ax.annotate("", xy=(slab_x - 0.35, y_bot + t_h), xytext=(slab_x - 0.35, y_bot + t_h + gap_h),
                arrowprops=dict(arrowstyle="<->", color="#222222"))
    ax.text(slab_x - 0.6, y_bot + t_h + gap_h / 2, "gap d", fontsize=14, ha="right", va="center")
    # "2r" spans an actual hole in the top slab (first hole centre + its offset).
    hole_c = slab_x + 0.9 + 0.35
    ax.annotate("", xy=(hole_c - 0.28, y_bot + 2 * t_h + gap_h + 0.25),
                xytext=(hole_c + 0.28, y_bot + 2 * t_h + gap_h + 0.25),
                arrowprops=dict(arrowstyle="<->", color="#222222"))
    ax.text(hole_c, y_bot + 2 * t_h + gap_h + 0.52, "2r", fontsize=14, ha="center")
    ax.plot([hole_c - 0.28] * 2, [y_bot + t_h + gap_h, y_bot + 2 * t_h + gap_h + 0.22],
            color="#888888", lw=0.9, ls=":")
    ax.plot([hole_c + 0.28] * 2, [y_bot + t_h + gap_h, y_bot + 2 * t_h + gap_h + 0.22],
            color="#888888", lw=0.9, ls=":")
    # illumination + measured quantity
    ax.annotate("", xy=(5.0, y_bot + 2 * t_h + gap_h + 0.1), xytext=(5.0, 9.4),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5))
    ax.text(5.25, 8.6, "RCP / LCP\nillumination", fontsize=12, color=GREEN, va="center")
    ax.annotate("", xy=(5.0, 0.7), xytext=(5.0, y_bot - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.5))
    ax.text(5.25, 1.05, r"$T_{RCP},\ T_{LCP}$", fontsize=13, color=PURPLE, va="center")
    ax.text(5.0, 0.05, r"$CD = (T_{RCP}-T_{LCP})/(T_{RCP}+T_{LCP})$",
            fontsize=12.5, ha="center", color="#222222")
    ax.set_title("Side view: the four sampled parameters")
    fig.suptitle("Device and design parameters  X = (θ, t, gap, r), lengths in units of a = 500 nm",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    return save(fig, "fig02_geometry.png")


# --------------------------------------------------------------------------- #
# 03 -- convergence study + cost wall                                          #
# --------------------------------------------------------------------------- #
def fig03_convergence():
    rows = []
    with open(os.path.join(CONVERGENCE, "convergence_results.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    labels = sorted({r["label"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for i, lab in enumerate(labels):
        sub = sorted((r for r in rows if r["label"] == lab), key=lambda r: int(r["N"]))
        n = [int(r["N"]) for r in sub]
        d = [float(r["delta_cd"]) for r in sub]
        keep = [(a, b) for a, b in zip(n, d) if np.isfinite(b) and b > 0]
        if keep:
            ax.plot([a for a, _ in keep], [b for _, b in keep], "o-",
                    color=cmap(i % 10), label=lab.replace("_", " "), alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel(r"truncation order  $N_m$")
    ax.set_ylabel(r"$\max_\lambda\,|CD_N - CD_{N+1}|$")
    ax.set_xticks([1, 2]), ax.set_xlim(0.85, 2.55)
    ax.axhspan(0.02, 0.05, color=ORANGE, alpha=0.15)
    ax.text(2.06, 0.028, "N=2 ↔ N=3\ndisagreement\n0.02–0.05", fontsize=10, color=VERMILLION)
    ax.set_title("Convergence: CD change per truncation order")
    ax.legend(fontsize=8.5, ncol=2, loc="lower left")

    # Per WAVELENGTH, not per spectrum: the convergence study ran a 9-point grid
    # (convergence_summary.json), so a per-spectrum number here would not be
    # comparable to the 26-point campaign costs quoted elsewhere.
    n_lam_study = 9
    ax = axes[1]
    times = {}
    for r in rows:
        t = float(r["runtime_s"])
        if np.isfinite(t) and t > 0:
            times.setdefault(int(r["N"]), []).append(t / n_lam_study)
    ns = sorted(times)
    med = [float(np.median(times[n])) for n in ns]
    ax.plot(ns, med, "o-", color=BLUE, label="measured (median, CPU)")
    i_ref = ns.index(2) if 2 in ns else 0
    ref = [med[i_ref] * ((2 * n + 1) ** 4) / ((2 * ns[i_ref] + 1) ** 4) for n in ns]
    ax.plot(ns, ref, "s--", color=GREY, label=r"$(2N_m+1)^4$ scaling")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel(r"truncation order  $N_m$")
    ax.set_ylabel("solver time per wavelength  (s)")
    ax.set_title("The cost wall")
    for n, t in zip(ns, med):
        ax.annotate(f"{t:.0f} s" if t >= 1 else f"{t:.1f} s", (n, t),
                    textcoords="offset points", xytext=(9, -13), fontsize=11)
    ax.legend(loc="upper left")
    fig.suptitle(r"Truncation $N_m$: accuracy improves, cost explodes as $(2N_m+1)^4$",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig03_convergence_cost_wall.png")


# --------------------------------------------------------------------------- #
# 04 -- GPU unlock                                                             #
# --------------------------------------------------------------------------- #
def fig04_gpu():
    b = BENCHMARKS
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    x = np.arange(2)
    cpu = [b["n2_cpu_s_per_wl"], b["n3_cpu_s_per_wl"]]
    gpu = [b["n2_gpu_s_per_wl"], b["n3_gpu_s_per_wl"]]
    ax.bar(x - 0.19, cpu, 0.36, label="CPU (complex128)", color=BLUE)
    ax.bar(x + 0.19, gpu, 0.36, label="GPU (complex64)", color=ORANGE)
    ax.set_yscale("log")
    ax.set_xticks(x, [r"$N_m=2$", r"$N_m=3$"])
    ax.set_ylabel("solver time per wavelength (s)")
    ax.set_title("GPU-resident RedhefferStar")
    for xi, c, g in zip(x, cpu, gpu):
        ax.annotate(f"{c:.1f}s", (xi - 0.19, c), ha="center", va="bottom", fontsize=10)
        ax.annotate(f"{g:.1f}s", (xi + 0.19, g), ha="center", va="bottom", fontsize=10)
        ax.annotate(f"{c/g:.1f}×", (xi, max(c, g) * 2.2), ha="center", fontsize=14,
                    fontweight="bold", color=GREEN)
    ax.set_ylim(top=max(cpu) * 12)
    ax.legend(loc="upper left")

    ax = axes[1]
    designs = 200
    days_cpu = b["n3_cpu_min_per_design"] * designs / 60 / 24
    days_gpu = b["n3_gpu_min_per_design"] * designs / 60 / 24
    bars = ax.bar(["CPU", "GPU"], [days_cpu, days_gpu], color=[BLUE, ORANGE], width=0.5)
    ax.bar_label(bars, labels=[f"{days_cpu:.1f} days", f"{days_gpu:.1f} days"],
                 padding=4, fontsize=13, fontweight="bold")
    ax.set_ylabel("wall-clock (days)")
    ax.set_ylim(0, days_cpu * 1.25)
    ax.set_title(f"Cost of a {designs}-design $N_m$=3 campaign\n(26 wavelengths per design)")
    ax.annotate("", xy=(0.82, days_gpu + 1.35), xytext=(0.18, days_cpu * 0.72),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(0.5, days_cpu * 0.88, "overnight-able", color=GREEN, fontsize=12.5,
            fontweight="bold", ha="center")
    fig.suptitle(r"Throughput unlock: 5.7× at $N_m$=3, accuracy gate |ΔCD| ~ 1e-6 vs CPU",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig04_gpu_unlock.png")


# --------------------------------------------------------------------------- #
# 05 -- sampling boxes and the CD signal they bought                           #
# --------------------------------------------------------------------------- #
def fig05_boxes():
    v0 = np.load(os.path.join(RAW, "v0_n120.npz"))
    hc = np.load(os.path.join(RAW, "highcd_n3_gpu_merged60.npz"))
    v0_peak, hc_peak = peak_abs_cd(v0["CD"]), peak_abs_cd(hc["CD"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    ax = axes[0]
    sc = ax.scatter(v0["X"][:, 0], v0["X"][:, 1], c=v0_peak, cmap="viridis",
                    s=42, edgecolor="white", linewidth=0.4, label="v0 (N=2, full box)")
    ax.scatter(hc["X"][:, 0], hc["X"][:, 1], facecolor="none", edgecolor=VERMILLION,
               s=52, linewidth=1.5, label="high-CD campaign (N=3)")
    # Labels sit above the (shared) top edge at opposite ends, clear of the points.
    for bounds, colour, lab, side in ((BOUNDS, GREY, "full box", "left"),
                                      (HIGH_CD_BOUNDS, VERMILLION, "high-CD box", "right")):
        (t0, t1), (h0, h1) = bounds["theta_deg"], bounds["thickness"]
        ax.add_patch(Rectangle((t0, h0), t1 - t0, h1 - h0, fill=False,
                               edgecolor=colour, lw=2, ls="--"))
        ax.text(t0 + 0.3 if side == "left" else t1 - 0.3, h1 + 0.004, lab, color=colour,
                fontsize=10.5, ha=side, va="bottom", fontweight="bold")
    ax.set_xlabel("twist θ (deg)"), ax.set_ylabel("thickness t  (units of a)")
    ax.set_ylim(0.085, 0.425)
    ax.set_title("Where the sampling budget went")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9, frameon=True)
    fig.colorbar(sc, ax=ax, label="peak |CD|")

    ax = axes[1]
    parts = ax.violinplot([v0_peak, hc_peak], showextrema=False, widths=0.75)
    for body, colour in zip(parts["bodies"], (GREY, VERMILLION)):
        body.set_facecolor(colour), body.set_alpha(0.35)
    for i, (vals, colour) in enumerate(((v0_peak, GREY), (hc_peak, VERMILLION)), start=1):
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.045, len(vals)),
                   vals, s=18, color=colour, alpha=0.65)
        ax.hlines(np.median(vals), i - 0.32, i + 0.32, color="black", lw=2.5)
        ax.annotate(f"median {np.median(vals):.3f}", (i + 0.36, np.median(vals)),
                    fontsize=11, va="center")
    ax.set_xticks([1, 2], [f"v0, full box\n(n={len(v0_peak)}, N=2)",
                           f"high-CD box\n(n={len(hc_peak)}, N=3)"])
    ax.set_ylabel("peak |CD| per design")
    # Ratio taken from the 3-dp medians shown on the plot, so the figure agrees
    # with the "x2.4" quoted in the experiment log.
    ratio = round(np.median(hc_peak), 3) / round(np.median(v0_peak), 3)
    ax.set_title(f"Focusing the box worked: ×{ratio:.1f} median signal")
    ax.axhline(1.0, color=GREEN, ls=":", lw=2)
    ax.text(1.5, 0.93, "device target |CD| → 1", color=GREEN, fontsize=11, ha="center")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return save(fig, "fig05_sampling_boxes.png")


# --------------------------------------------------------------------------- #
# 06 -- cross-validated learnability                                           #
# --------------------------------------------------------------------------- #
def fig06_cv():
    with open(os.path.join(ARTIFACTS, "cv_report_highcd_n3_gpu_merged60.json")) as fh:
        main = json.load(fh)["rows"]
    with open(os.path.join(ARTIFACTS, "cv_report_resonance_pilot20.json")) as fh:
        pilot = json.load(fh)["rows"]

    order = ["cd", "delta_t", "both_t"]
    pretty = {"cd": "CD spectrum", "delta_t": r"$\Delta T = T_{RCP}-T_{LCP}$",
              "both_t": "transmission\n$T_{RCP}, T_{LCP}$"}
    rows = {r["target"]: r for r in main}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    x = np.arange(len(order))
    m = [rows[t]["model_r2_mean"] for t in order]
    ms = [rows[t]["model_r2_std"] for t in order]
    b = [rows[t]["baseline_r2_mean"] for t in order]
    bs = [rows[t]["baseline_r2_std"] for t in order]
    ax.bar(x - 0.19, m, 0.36, yerr=ms, capsize=5, color=BLUE,
           label="Gaussian-process surrogate")
    ax.bar(x + 0.19, b, 0.36, yerr=bs, capsize=5, color=GREY, alpha=0.55,
           label="mean-predictor baseline")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [pretty[t] for t in order])
    ax.set_ylabel(r"cross-validated test $R^2$")
    ax.set_ylim(-0.42, 0.92)
    ax.set_title(r"n=60, $N_m$=3 high-CD box" "\n(5-fold × 10 repeats = 50 trials)")
    ax.legend(loc="upper left")
    for xi in (0, 1):
        ax.annotate("tied with\nbaseline", (xi, 0.05), ha="center", va="bottom",
                    color=VERMILLION, fontsize=11.5, fontweight="bold")
    ax.annotate("clearly\nlearnable", (2, rows["both_t"]["model_r2_mean"] + 0.10),
                ha="center", va="bottom", color=GREEN, fontsize=11.5, fontweight="bold")

    ax = axes[1]
    mlp = next(r for r in pilot if r["model"] == "mlp")
    entries = [("CD spectrum\nGP, n=60", rows["cd"]["model_r2_mean"], rows["cd"]["model_r2_std"], VERMILLION),
               ("resonance\nfeats, MLP\nn=20", mlp["model_r2_mean"], mlp["model_r2_std"], ORANGE),
               ("resonance\nfeats, GP\nn=20", RESONANCE_CV_GP["model_r2_mean"],
                RESONANCE_CV_GP["model_r2_std"], ORANGE)]
    for i, (lab, mu, sd, colour) in enumerate(entries):
        ax.errorbar(i, mu, yerr=sd, fmt="o", color=colour, capsize=9, markersize=10, lw=2.5)
        ax.annotate(f"±{sd:.2f}", (i + 0.14, mu), fontsize=11.5, va="center", color=colour,
                    fontweight="bold")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(3), [e[0] for e in entries], fontsize=10.5)
    ax.set_xlim(-0.55, 2.75)
    ax.set_ylim(-1.5, 0.75)
    ax.set_ylabel(r"cross-validated test $R^2$")
    ax.set_title("A well-powered null vs.\nan inconclusive one")
    ax.text(-0.45, 0.42, "at n=20 the error bars are 8–11× wider:\ncannot tell 'no signal' apart from\n'signal too weak to detect'",
            fontsize=10.5, color=ORANGE, ha="left", va="top")
    fig.tight_layout()
    return save(fig, "fig06_cv_learnability.png")


# --------------------------------------------------------------------------- #
# 07 -- learning curves                                                        #
# --------------------------------------------------------------------------- #
def fig07_learning_curves():
    with open(os.path.join(ARTIFACTS, "forward_surrogate_both_t_gp_report.json")) as fh:
        rep_t = json.load(fh)
    with open(os.path.join(ARTIFACTS, "forward_surrogate_cd_gp_report.json")) as fh:
        rep_cd = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, rep, colour, title in (
        (axes[0], rep_t, GREEN, "Transmission: more data helps"),
        (axes[1], rep_cd, VERMILLION, "CD: flat — more data does not"),
    ):
        lc = rep["learning_curve"]
        sizes, rmse = np.array(lc["sizes"]), np.array(lc["val_rmse"])
        sd = np.array(lc.get("val_rmse_std", np.zeros_like(rmse)))
        ax.errorbar(sizes, rmse, yerr=sd, fmt="o-", color=colour, capsize=5,
                    label="GP surrogate")
        base = rep["baseline_test"]["rmse"]
        ax.axhline(base, color=GREY, ls="--", lw=2, label=f"mean-predictor baseline ({base:.3f})")
        drop = 100 * (rmse[-1] / rmse[0] - 1)
        ax.set_xlabel("training designs")
        ax.set_ylabel("validation RMSE")
        ax.set_title(f"{title}\n({rep['shard']}, n={rep['n']}, GP)", fontsize=13)
        ax.annotate(f"{rmse[0]:.3f} → {rmse[-1]:.3f}   ({drop:+.0f}% RMSE)",
                    (sizes[-1], rmse[-1]), textcoords="offset points", xytext=(-10, 20),
                    ha="right", fontsize=12, color=colour, fontweight="bold")
        ax.set_ylim(0, max(rmse.max(), base) * 1.35)
        ax.legend(loc="lower left", fontsize=10.5)
    fig.suptitle("Learning curves: the shape, not the level, is the diagnosis", fontsize=14)
    fig.tight_layout()
    return save(fig, "fig07_learning_curves.png")


# --------------------------------------------------------------------------- #
# 08 / 09 -- spectral aliasing                                                 #
# --------------------------------------------------------------------------- #
def _dense_data():
    d = np.load(os.path.join(RAW, "dense_rescan_highcd_n3_gpu_merged60.npz"))
    return d, [int(i) for i in d["indices"]]


def fig08_aliasing():
    d, indices = _dense_data()
    dense_wl, coarse_wl = d["dense_wavelengths_nm"], d["coarse_wavelengths_nm"]
    labels = {38: "highest peak |CD|", 26: "mid peak |CD|", 48: "smallest gap"}

    fig, axes = plt.subplots(1, len(indices), figsize=(13.5, 4.6), sharex=True)
    for ax, idx in zip(np.atleast_1d(axes), indices):
        ax.plot(dense_wl, d[f"design{idx}_dense_cd"], color=BLUE, lw=1.8,
                label="dense scan (151 pts, 1.7 nm)")
        ax.plot(dense_wl, d[f"design{idx}_coarse_interp_on_dense"], color=GREY, lw=1.4,
                ls="--", label="26-pt grid, interpolated")
        ax.plot(coarse_wl, d[f"design{idx}_coarse_cd"], "o", color=VERMILLION,
                markersize=5, label="26-pt campaign samples")
        ax.set_title(f"design {idx} — {labels.get(idx, '')}", fontsize=13)
        ax.set_xlabel("wavelength (nm)")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    np.atleast_1d(axes)[0].set_ylabel("CD")
    np.atleast_1d(axes)[0].legend(fontsize=9.5, loc="upper right", framealpha=0.9, frameon=True)
    fig.suptitle("The 26-point grid was aliasing the CD spectrum it was meant to measure",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig08_aliasing_spectra.png")


def fig09_aliasing_stats():
    d, indices = _dense_data()
    ratios, dense_peaks, coarse_peaks = [], [], []
    for idx in indices:
        dense = d[f"design{idx}_dense_cd"]
        interp = d[f"design{idx}_coarse_interp_on_dense"]
        ratios.append(float(np.sqrt(np.mean((dense - interp) ** 2)) / np.std(dense)))
        dense_peaks.append(float(np.max(np.abs(dense))))
        # Peak read off the INTERPOLATED coarse curve -- what someone reading the
        # campaign spectrum would conclude (matches the experiment log's numbers).
        coarse_peaks.append(float(np.max(np.abs(interp))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    x = np.arange(len(indices))
    ax = axes[0]
    bars = ax.bar(x, ratios, 0.5, color=VERMILLION)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12, fontweight="bold")
    ax.set_xticks(x, [f"design {i}" for i in indices])
    ax.set_ylabel("RMS(dense − coarse) / std(dense)")
    ax.set_ylim(0, max(ratios) * 1.3)
    ax.set_title("Fraction of the true CD variation\nthe campaign grid missed")

    ax = axes[1]
    ax.bar(x - 0.19, dense_peaks, 0.36, color=BLUE, label="true (dense scan)")
    ax.bar(x + 0.19, coarse_peaks, 0.36, color=GREY, label="what the 26-pt grid implied")
    for xi, dp, cp in zip(x, dense_peaks, coarse_peaks):
        ax.annotate(f"{dp:.3f}", (xi - 0.19, dp), ha="center", va="bottom", fontsize=10)
        ax.annotate(f"{cp:.3f}", (xi + 0.19, cp), ha="center", va="bottom", fontsize=10)
        err = 100 * (cp - dp) / dp
        if abs(err) > 10:
            ax.annotate(f"{err:+.0f}%", (xi + 0.19, cp + 0.03), ha="center",
                        fontsize=12, color=VERMILLION, fontweight="bold")
    ax.set_xticks(x, [f"design {i}" for i in indices])
    ax.set_ylabel("peak |CD|")
    ax.set_ylim(0, max(dense_peaks) * 1.35)
    ax.set_title("The labels themselves were wrong")
    ax.legend(loc="upper right", fontsize=10.5)
    fig.suptitle("Aliasing quantified: 41–58% of the CD signal was invisible to the grid",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig09_aliasing_quantified.png")


# --------------------------------------------------------------------------- #
# 10 -- Fano fits on a dense spectrum                                          #
# --------------------------------------------------------------------------- #
def fig10_fano(idx: int = 38):
    d, _ = _dense_data()
    lam = d["dense_wavelengths_nm"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    from data_generation.resonances import fano_lineshape
    cmap = plt.get_cmap("tab10")
    for ax, pol in ((axes[0], "T_RCP"), (axes[1], "T_LCP")):
        y = d[f"design{idx}_dense_{pol}"]
        ax.plot(lam, y, color="#333333", lw=2.0, label=f"{pol} (dense scan, 151 pts)", zorder=3)
        fits = [f for f in fit_spectrum(lam, y, window_nm=60.0) if f.success and f.r2 > 0.9]
        for j, f in enumerate(fits):
            half = f.window_nm / 2
            sel = (lam >= f.lam0 - half) & (lam <= f.lam0 + half)
            ax.plot(lam[sel], fano_lineshape(lam[sel], f.lam0, f.gamma, f.q, f.amp, f.bg),
                    "--", lw=2.4, color=cmap(j % 10), zorder=4)
            ax.plot([f.lam0], [y[np.argmin(np.abs(lam - f.lam0))]], "v", color=cmap(j % 10),
                    markersize=7, zorder=5)
        qs = [f.Q for f in fits]
        ax.set_xlabel("wavelength (nm)")
        ax.set_title(f"design {idx}: {pol}", fontsize=13.5)
        ax.text(0.03, 0.05, f"{len(fits)} resonances fitted (r² > 0.9)\n"
                            f"Q = {min(qs):.0f} – {max(qs):.0f}",
                transform=ax.transAxes, fontsize=11.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC"))
        ax.legend(loc="upper right", fontsize=10.5)
        # Crop to the data's own range: a fitted lineshape is only meaningful
        # inside its window, and its tails otherwise stretch the axes.
        pad = 0.06 * (y.max() - y.min())
        ax.set_ylim(y.min() - pad, y.max() + pad)
    axes[0].set_ylabel("transmission")
    fig.suptitle("Fitting T_RCP and T_LCP as Fano lineshapes: (λ₀, Q, amplitude) instead of grid points",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig10_fano_fits.png")


# --------------------------------------------------------------------------- #
# 11 -- iteration timeline and the open fork                                   #
# --------------------------------------------------------------------------- #
def fig11_timeline():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13), ax.set_ylim(0, 6), ax.axis("off"), ax.grid(False)

    steps = [
        ("Jun 30", "convergence\nstudy", "N=3 costs\n~26× N=2", GREEN),
        ("Jul 02", "v0 dataset\nn=120, N=2", "T learnable,\nCD not", ORANGE),
        ("Jul 02", "GPU unlock\ngmatmul", "5.7× at N=3", GREEN),
        ("Jul 03", "high-CD box\nn=60, N=3", "signal ×2.4,\nCD still not", ORANGE),
        ("Jul 03", "repeated CV\n+ dense rescan", "null is real\n→ aliasing", VERMILLION),
        ("Jul 04", "Fano fitting\nmodule", "validated,\n7 bugs fixed", GREEN),
        ("Jul 05", "densify pilot\nn=20 features", "inconclusive", ORANGE),
    ]
    y = 4.35
    ax.plot([0.35, 12.4], [y, y], color="#BBBBBB", lw=3, zorder=0)
    for i, (date, title, outcome, colour) in enumerate(steps):
        x = 0.6 + i * 1.94
        ax.scatter([x], [y], s=190, color=colour, zorder=3, edgecolor="white", linewidth=2)
        ax.text(x, y + 0.95, title, ha="center", fontsize=11, fontweight="bold", linespacing=1.3)
        ax.text(x, y + 0.45, date, ha="center", fontsize=10, color="#666666")
        # Stagger the outcome captions: neighbouring labels are wider than the
        # tick spacing and would otherwise run into each other.
        ax.text(x, y - (0.35 if i % 2 == 0 else 1.05), outcome, ha="center", va="top",
                fontsize=10, color=colour, linespacing=1.3, fontweight="bold")

    ax.text(6.4, 2.45, "we are here  →  the fork", ha="center", fontsize=14, fontweight="bold")
    options = [
        ("A. Densify ~40 more designs", "≈ 42 GPU-hours; shrinks the\nerror bars, gives a real answer", GREEN),
        ("B. Change representation again", "more features / larger K\nat the same n = 20", ORANGE),
        ("C. Deprioritize the surrogate", "keep Q-extraction; pivot to\nadjoint / gradient inverse design", BLUE),
    ]
    for i, (title, sub, colour) in enumerate(options):
        x = 0.5 + i * 4.2
        box(ax, x, 0.5, 3.7, 1.55, f"{title}\n\n{sub}", face=colour + "1A", edge=colour, fontsize=10.5)
    ax.set_title("Seven iterations, three honest outcomes — and the decision this meeting is for", pad=12)
    return save(fig, "fig11_timeline_fork.png")


# --------------------------------------------------------------------------- #
# Stage C -- the peak-centred narrow-window representation (EXPERIMENTS Sec 6h/6i)
#
# These read from whatever peak shards are on disk, so re-running after another
# campaign shard lands refreshes every number without editing this file.
# --------------------------------------------------------------------------- #

PEAK_SHARDS = ["peak_known8", "peak_pilot25", "peak_batch2a", "peak_batch2b",
               "peak_redo1"]  # redo LAST: dedupe keeps the last-listed occurrence


def _peak_merged():
    """Merged accepted designs across every peak shard present on disk."""
    from scripts.peak_report import load_shards
    have = [s for s in PEAK_SHARDS
            if os.path.exists(os.path.join(RAW, f"{s}.npz"))]
    return load_shards(have), have


def fig12_window_concept():
    """Why the window is peak-centred: peaks are spread across the whole band."""
    D, _ = _peak_merged()
    fp, u = D["f_peak"], D["u"]
    a_nm = 500.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    counts, _, _ = ax1.hist(a_nm / fp, bins=14, color=BLUE, alpha=0.8, edgecolor="white")
    # Headroom so the annotation never collides with the tallest bar.
    ax1.set_ylim(0, counts.max() * 1.32)
    ax1.set(xlabel="wavelength of each design's CD peak (nm)", ylabel="designs",
            title=f"Peaks are spread over ~{a_nm/fp.max():.0f}–{a_nm/fp.min():.0f} nm")
    ax1.text(0.5, 0.94, "no single fixed window can catch them all",
             transform=ax1.transAxes, ha="center", va="top", fontsize=11,
             color=VERMILLION,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="none", alpha=0.85))

    # Every design's CD on the shared relative axis: aligned by construction.
    for cd in D["window_cd"]:
        ax2.plot(100 * u, cd, color=BLUE, alpha=0.28, lw=1.0)
    ax2.axvline(0, color=VERMILLION, lw=1.6, ls="--")
    ax2.text(0.4, 0.96, "u = 0: each design's own peak", transform=ax2.transAxes,
             fontsize=10.5, color=VERMILLION, va="top")
    ax2.set(xlabel=r"relative frequency  $u = \tilde{f}/\tilde{f}_{\rm peak} - 1$  (%)",
            ylabel="CD", title=f"…so each is measured on its own window (n={len(fp)})")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.5)

    fig.suptitle("Stage C: a narrow window centred on each design's own CD peak", fontsize=14)
    fig.tight_layout()
    return save(fig, "fig12_window_concept.png")


def fig13_dealiasing_recovery():
    """What the coarse pass reports vs what the resolved window measures."""
    D, _ = _peak_merged()
    loc, win = D["locate_peak_abs_cd"], D["window_peak_abs_cd"]
    ok = np.isfinite(loc) & (loc > 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    lim = [0, max(win.max(), loc[ok].max()) * 1.08]
    ax1.plot(lim, lim, color=GREY, ls="--", lw=1.4, label="agreement")
    ax1.scatter(loc[ok], win[ok], s=46, color=BLUE, alpha=0.85, edgecolor="white", zorder=3)
    ax1.set(xlim=lim, ylim=lim, xlabel="peak |CD| from the coarse locate pass",
            ylabel="peak |CD| from the resolved window",
            title="Points above the line: the coarse grid under-read")
    ax1.legend(loc="lower right")

    ratio = win[ok] / loc[ok]
    ax2.hist(ratio, bins=14, color=GREEN, alpha=0.85, edgecolor="white")
    ax2.axvline(1.0, color=GREY, ls="--", lw=1.4)
    ax2.axvline(np.median(ratio), color=VERMILLION, lw=2.0,
                label=f"median {np.median(ratio):.2f}")
    ax2.set(xlabel="resolved / coarse peak |CD|", ylabel="designs",
            title=f"Under-read in {100*(ratio > 1.05).mean():.0f}% of designs "
                  f"(max {ratio.max():.2f}×)")
    ax2.legend()

    fig.suptitle("De-aliasing: the resolved window recovers CD the coarse grid missed",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig13_dealiasing_recovery.png")


def _two_stage_rows(target: str):
    """Paired CV of flat / conditioned / baseline on the merged peak set."""
    import warnings
    from models.forward_surrogate import make_fit
    from models.two_stage_surrogate import cv_two_stage

    D, _ = _peak_merged()
    X = D["X"].astype(float)
    m, s = X.mean(0), X.std(0)
    s[s < 1e-12] = 1.0
    Xn = ((X - m) / s).astype(np.float32)
    Y = (D["window_cd"] if target == "rel_cd"
         else np.concatenate([D["window_T_RCP"], D["window_T_LCP"]], axis=1)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cv_two_stage(Xn, D["f_peak"].astype(np.float32), Y, make_fit("gp"), 5, 10, 0), len(X)


def fig14_cd_conditioned():
    """The headline: CD is learnable once conditioned on where the peak sits."""
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, (target, label) in zip(axes, [("rel_cd", "CD lineshape"),
                                          ("both_t", "transmission (control)")]):
        r, n = _two_stage_rows(target)
        names = ["mean-predictor\nbaseline", "flat\nX → lineshape",
                 "conditioned on\nmeasured $\\tilde{f}_{peak}$"]
        keys = ["baseline", "flat", "oracle"]
        means = [r[k].mean() for k in keys]
        sds = [r[k].std() for k in keys]
        colours = [GREY, ORANGE, GREEN]

        bars = ax.bar(names, means, yerr=sds, capsize=5, color=colours,
                      alpha=0.9, edgecolor="white")
        ax.axhline(0, color="black", lw=1.0)
        for b, mval in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2,
                    mval + (0.03 if mval >= 0 else -0.09),
                    f"{mval:+.3f}", ha="center", fontsize=11, fontweight="bold")
        d = r["oracle"] - r["baseline"]
        p = stats.ttest_1samp(d, 0)[1]
        ax.set_title(f"{label}  (n={n})\nconditioned vs baseline: "
                     f"{d.mean():+.3f}, paired p = {p:.1e}", fontsize=12.5)
        ax.set_ylabel("CV $R^2$  (50 paired folds)")

    fig.suptitle("Knowing where the resonance sits is what makes CD learnable",
                 fontsize=14)
    fig.tight_layout()
    return save(fig, "fig14_cd_conditioned.png")


def fig15_deployable_pipeline():
    """The payoff: f_peak is measured cheaply, not predicted."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.4),
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 3.2); ax1.axis("off"); ax1.grid(False)

    box(ax1, 0.2, 1.15, 1.9, 1.0, "design\n(θ, t, gap, r)", face=GREY + "1A", edge=GREY)
    box(ax1, 2.6, 1.15, 2.5, 1.0, "locate pass\nN=2, 81 pts\n~2.5 min",
        face=BLUE + "1A", edge=BLUE, fontsize=10.5)
    box(ax1, 5.6, 1.15, 2.1, 1.0, "surrogate\n(X, $\\tilde{f}_{peak}$)\ninstant",
        face=GREEN + "1A", edge=GREEN, fontsize=10.5)
    box(ax1, 8.1, 1.15, 1.7, 1.0, "resolved\nlineshape", face=GREEN + "1A", edge=GREEN,
        fontsize=10.5)
    for a, b in [((2.1, 1.65), (2.6, 1.65)), ((5.1, 1.65), (5.6, 1.65)),
                 ((7.7, 1.65), (8.1, 1.65))]:
        arrow(ax1, a, b)
    box(ax1, 2.6, 0.05, 5.1, 0.8, "replaces the N=3 window pass  (~23 min)",
        face=VERMILLION + "1A", edge=VERMILLION, fontsize=10.5)
    ax1.text(5.0, 2.55, "Deployable pipeline", ha="center", fontsize=13.5, fontweight="bold")

    stages = ["full two-pass\nmeasurement", "locate + surrogate"]
    mins = [2.5 + 23.0, 2.5]
    bars = ax2.bar(stages, mins, color=[GREY, GREEN], alpha=0.9, edgecolor="white", width=0.55)
    for b, v in zip(bars, mins):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f} min",
                 ha="center", fontsize=12, fontweight="bold")
    ax2.set(ylabel="minutes per design", ylim=(0, 30),
            title=f"{mins[0]/mins[1]:.0f}× cheaper per design")

    fig.suptitle("$\\tilde{f}_{peak}$ is measured in ~2.5 min, not predicted — "
                 "so the conditioned model is deployable", fontsize=13.5)
    fig.tight_layout()
    return save(fig, "fig15_deployable_pipeline.png")


def fig16_learning_curve():
    """Is CD's accuracy actually improving with n, or has it plateaued?

    The decision figure: every earlier campaign produced a FLAT learning curve
    at baseline, which is what "more data will not help" looks like. This plots
    the same quantity for the conditioned model.
    """
    import warnings
    from models.forward_surrogate import make_fit
    from models.two_stage_surrogate import cv_two_stage

    D, _ = _peak_merged()
    X = D["X"].astype(float)
    m, s = X.mean(0), X.std(0)
    s[s < 1e-12] = 1.0
    Xn = ((X - m) / s).astype(np.float32)
    fp = D["f_peak"].astype(np.float32)
    Y = D["window_cd"].astype(np.float32)
    n_all = len(fp)

    # Sizes spaced so no two points are near-duplicate subsamples of the same
    # pool: at n_all-3 vs n_all the two draws share almost every design, so any
    # difference between them is noise that reads as a real wobble on the plot.
    sizes = [x for x in (20, 30, 40, 50, 60) if x <= n_all - 6]
    sizes.append(n_all)
    rng = np.random.default_rng(0)

    cond_mu, cond_sd, base_mu = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in sizes:
            reps, bases = [], []
            n_draws = 1 if n == n_all else 5   # the full set has only one draw
            for rep in range(n_draws):
                idx = rng.choice(n_all, n, replace=False) if n < n_all else np.arange(n_all)
                r = cv_two_stage(Xn[idx], fp[idx], Y[idx], make_fit("gp"), 5, 5, rep)
                reps.append(r["oracle"].mean())
                bases.append(r["baseline"].mean())
            cond_mu.append(np.mean(reps)); cond_sd.append(np.std(reps))
            base_mu.append(np.mean(bases))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.errorbar(sizes, cond_mu, yerr=cond_sd, marker="o", color=GREEN, capsize=4,
                label="CD conditioned on measured $\\tilde{f}_{peak}$")
    ax.plot(sizes, base_mu, marker="s", ls="--", color=GREY, label="mean-predictor baseline")
    ax.axhline(0, color="black", lw=0.9)
    ax.set(xlabel="training-set size (designs)", ylabel="CV $R^2$",
           title=f"CD accuracy vs data: {cond_mu[0]:+.2f} at n={sizes[0]} "
                 f"→ {cond_mu[-1]:+.2f} at n={sizes[-1]}")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04,
            "every earlier representation gave a FLAT\ncurve pinned at baseline",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color=VERMILLION)
    fig.tight_layout()
    return save(fig, "fig16_cd_learning_curve.png")


FIGURES = {
    1: fig01_pipeline, 2: fig02_geometry, 3: fig03_convergence, 4: fig04_gpu,
    5: fig05_boxes, 6: fig06_cv, 7: fig07_learning_curves, 8: fig08_aliasing,
    9: fig09_aliasing_stats, 10: fig10_fano, 11: fig11_timeline,
    12: fig12_window_concept, 13: fig13_dealiasing_recovery,
    14: fig14_cd_conditioned, 15: fig15_deployable_pipeline,
    16: fig16_learning_curve,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, nargs="+", choices=sorted(FIGURES),
                   help="regenerate only these figure numbers")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    if args.out_dir:
        globals()["OUT_DIR"] = args.out_dir
    style()
    wanted = args.only or sorted(FIGURES)
    for n in wanted:
        print(f"[fig{n:02d}] {FIGURES[n].__name__}")
        FIGURES[n]()
    print(f"\n{len(wanted)} figure(s) in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
