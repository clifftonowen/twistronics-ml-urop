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
# (+0.37) correlate most strongly, radius weakly (+0.22), gap weakly and
# inversely (-0.21). Top-15 CD designs in v0 clustered at theta 15-30
# (med 25), t 0.25-0.40 (med 0.31), gap 0-0.25 (med 0.18), r 0.15-0.37
# (med 0.22). Narrowing BOUNDS to this box concentrates sampling density in
# the strong-CD region instead of spending most of the LHS budget on
# near-achiral (weak-CD) structures, which is what v0's uniform box did.
HIGH_CD_BOUNDS: dict[str, tuple[float, float]] = {
    "theta_deg": (15.0, 30.0),
    "thickness": (0.25, 0.40),
    "gap": (0.00, 0.25),
    "radius": (0.15, 0.35),
}

# Named presets selectable via generate_dataset.py --box.
BOX_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "full": BOUNDS,
    "high_cd": HIGH_CD_BOUNDS,
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


def params_to_matrix(params: list[DesignParams]) -> np.ndarray:
    """Stack a list of DesignParams into an (n, n_params) float matrix X."""
    return np.vstack([p.as_vector() for p in params])
