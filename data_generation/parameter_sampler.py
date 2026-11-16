"""Sample structural design parameters X for the dataset.

The model input X for a twisted bilayer PhC:
  * twist angle      theta  [degrees]
  * slab thickness   t      [units of a]
  * interlayer gap   d      [units of a]   (primary chirality knob)
  * hole radius      r      [units of a]

For v1 the materials and lattice constant are held fixed (recorded in
metadata, not sampled) so the dataset varies only the geometric knobs. Adding
material / lattice-constant dimensions later is a matter of widening BOUNDS.

Sampling uses a Latin Hypercube (scipy.stats.qmc) for even coverage of the
4-D box with few points -- far better than i.i.d. uniform at small N. Each
sample is an independent *structure*; the train/val/test split downstream is
by structure, which is automatic here since no two rows
share a sweep.

DESIGN-RANGE RATIONALE (defaults):
  * theta 5-30 deg: large twists keep the moire supercell small and the solver
    cheap (cost ~ (2*N_m+1)^4). Sub-degree twists are the expensive regime and
    are deliberately excluded from v1.
  * t 0.10-0.40 a: membrane thickness; sets the background-transmission level
    that the CD recipe (one mode near T~0/1) rides on.
  * d 0.00-0.50 a: interlayer gap. Too large -> evanescent coupling dies and CD
    vanishes; kept variable because the chirality literature treats it as the
    main CD control.
  * r 0.10-0.45 a: hole radius; must stay < 0.5 to fit one hole per cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import qmc

# name -> (low, high). Order here defines the column order of X.
BOUNDS: dict[str, tuple[float, float]] = {
    "theta_deg": (5.0, 30.0),
    "thickness": (0.10, 0.40),
    "gap": (0.00, 0.50),
    "radius": (0.10, 0.45),
}

# High-CD focused box (EXPERIMENTS.md Sec 5): mined from v0_n120 by rank
# correlation of each parameter with peak |CD| -- thickness (+0.43) and twist
# (+0.37) appeared to correlate most strongly, radius weakly (+0.22), gap
# weakly and inversely (-0.21). Top-15 CD designs in v0 clustered at theta
# 15-30 (med 25), t 0.25-0.40 (med 0.31), gap 0-0.25 (med 0.18), r 0.15-0.37
# (med 0.22).
#
# CAVEAT, established later and important: those correlations were computed on
# labels from the 26-point/600-850nm grid, which was subsequently shown to
# alias the CD spectrum (EXPERIMENTS.md Sec 6c). Recomputed against the 20
# designs that now have properly resolved dense scans, the twist correlation
# does not survive -- Spearman rho(theta, peak|CD|) = +0.05, p = 0.84, versus
# the +0.37 quoted above, and the single strongest design in that set sits at
# theta = 20.4. Thickness (+0.32) and gap (+0.30) are nominally positive but
# also not significant at n = 20. Treat this box as a historical artifact that
# did raise measured CD in practice (median peak|CD| 0.078 -> 0.191), not as
# evidence about where CD actually lives.
HIGH_CD_BOUNDS: dict[str, tuple[float, float]] = {
    "theta_deg": (15.0, 30.0),
    "thickness": (0.25, 0.40),
    "gap": (0.00, 0.25),
    "radius": (0.15, 0.35),
}

# Box for the peak-centred narrow-window campaign. Narrowed around the six
# designs with theta >= 25 that have resolved dense scans and peak|CD| >= 0.24
# (spans: t 0.296-0.398, gap 0.073-0.235, r 0.184-0.329), padded slightly so
# the seeds are not on the boundary.
#
# The theta 25-30 restriction is a LABEL-QUALITY and DENSITY choice, not a
# CD-maximizing one -- see the caveat on HIGH_CD_BOUNDS above. At a fixed
# truncation N_m = 3, a larger twist gives a smaller moire supercell and so
# needs fewer harmonics to converge, making labels in this corner the most
# trustworthy available. Narrowing all four dimensions cuts the box volume to
# ~21% of the high_cd box, roughly a 4.8x gain in sampling density at equal n,
# which is the binding constraint on learnability.
PEAK_BOUNDS: dict[str, tuple[float, float]] = {
    "theta_deg": (25.0, 30.0),
    "thickness": (0.27, 0.40),
    "gap": (0.05, 0.25),
    "radius": (0.16, 0.34),
}

# Gap near a vacuum quarter-wave (supervisor feedback, 2026-09-18): the gap
# layer is vacuum (simulate_spectra.py builds it with eps=None), so d = lambda/4
# means d/a = 1/(4*f_tilde). Across the locate band f_tilde in [0.60, 0.92],
# that is d/a in [0.27, 0.42] -- all 71 PEAK_BOUNDS designs sit at d/a <= 0.235,
# so this box is deliberately outside anything measured so far. theta/t/r are
# left identical to PEAK_BOUNDS so gap is the only variable that changes
# relative to the existing n=71 set.
QUARTER_WAVE_BOUNDS: dict[str, tuple[float, float]] = {
    "theta_deg": (25.0, 30.0),
    "thickness": (0.27, 0.40),
    "gap": (0.27, 0.42),
    "radius": (0.16, 0.34),
}

# Named presets selectable via generate_dataset.py --box.
BOX_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "full": BOUNDS,
    "high_cd": HIGH_CD_BOUNDS,
    "peak_box": PEAK_BOUNDS,
    "quarter_wave": QUARTER_WAVE_BOUNDS,
}

PARAM_NAMES = list(BOUNDS.keys())


@dataclass(frozen=True)
class DesignParams:
    """One structure. Geometric lengths are in units of the lattice constant a."""

    theta_deg: float
    thickness: float
    gap: float
    radius: float

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in PARAM_NAMES], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def sample_params(
    n: int,
    bounds: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> list[DesignParams]:
    """Draw `n` design points via Latin Hypercube sampling within `bounds`."""
    bounds = bounds or BOUNDS
    names = list(bounds.keys())
    lows = np.array([bounds[k][0] for k in names])
    highs = np.array([bounds[k][1] for k in names])

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n)              # (n, d) in [0, 1)
    scaled = qmc.scale(unit, lows, highs)  # (n, d) in [low, high)

    return [DesignParams(**dict(zip(names, row))) for row in scaled]


def sample_perturbed(
    seeds: list[DesignParams],
    n: int,
    bounds: dict[str, tuple[float, float]] | None = None,
    sigma_frac: float = 0.15,
    seed: int | None = None,
) -> list[DesignParams]:
    """Draw `n` designs by jittering around known-good `seeds`.

    Complements `sample_params`: LHS spreads a fixed budget evenly, which is
    what you want for coverage, while this concentrates it near structures
    already measured to have strong CD -- the region an inverse-design search
    will actually operate in, and where the response is most worth resolving.

    Per-dimension sigma is `sigma_frac` of that dimension's box WIDTH rather
    than of the value, so the jitter is comparable across parameters that live
    on different scales (theta in degrees, the rest in units of a).

    Seeds are cycled round-robin, not sampled at random, so a small `n` cannot
    happen to cluster on one seed and leave the others unexplored.

    Draws are clipped to `bounds`. Clipping (rather than rejecting) means seeds
    near a face pile a little probability mass onto it; that is deliberate --
    the box faces are chosen conventions, not physical walls, and rejection
    would silently thin the sample near exactly the good designs this is meant
    to concentrate on.
    """
    if not seeds:
        raise ValueError("need at least one seed design")
    bounds = bounds or PEAK_BOUNDS
    names = list(bounds.keys())
    lows = np.array([bounds[k][0] for k in names])
    highs = np.array([bounds[k][1] for k in names])
    sigma = sigma_frac * (highs - lows)

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = np.array([getattr(seeds[i % len(seeds)], k) for k in names], dtype=float)
        draw = np.clip(base + rng.normal(0.0, sigma), lows, highs)
        out.append(DesignParams(**dict(zip(names, draw))))
    return out


def params_to_matrix(params: list[DesignParams]) -> np.ndarray:
    """Stack a list of DesignParams into an (n, n_params) float matrix X."""
    return np.vstack([p.as_vector() for p in params])
