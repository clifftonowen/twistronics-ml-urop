"""Build real-space relative-permittivity maps for a single PhC slab layer.

A photonic-crystal slab here is a high-index dielectric membrane with a square
lattice of circular holes (one hole per unit cell). The RCWA solver expands
this map in a Fourier (plane-wave) basis via `convmat2D`, so we only need to
hand it a 2D grid of epsilon_r sampled over one unit cell.

Conventions (match the rcwa4d example notebooks and CLAUDE.md section 7):
  * The unit cell spans [-0.5, 0.5] x [-0.5, 0.5] in units of the lattice
    constant `a`. ALL lengths (hole radius, later the slab thickness and gap)
    are expressed as fractions of `a`. The solver runs with a = 1 (normalized),
    and physical wavelengths enter only through the normalized frequency
    f~ = a/lambda. Keep `a` (in nm) in the dataset metadata to recover nm.
  * Grid index [row=y, col=x], shape (resolution, resolution).
"""

from __future__ import annotations

import numpy as np

from .materials import eps_of


def square_lattice_holes(
    eps_slab: float,
    radius: float,
    eps_hole: float = 1.0,
    resolution: int = 512,
) -> np.ndarray:
    """One unit cell: high-index background with a central circular hole.

    Args:
        eps_slab: relative permittivity of the slab material.
        radius: hole radius in units of the lattice constant `a` (0 < r < 0.5).
        eps_hole: permittivity inside the hole (1.0 = air; use SiO2 for a
            spacer-filled hole).
        resolution: pixels per side of the unit-cell grid. 512 is plenty for
            the low diffraction orders we keep; higher only matters near sharp
            edges. The example notebooks use 1000.

    Returns:
        (resolution, resolution) float array of epsilon_r.
    """
    if not (0.0 < radius < 0.5):
        raise ValueError(f"radius must be in (0, 0.5) units of a; got {radius}")

    eps = np.full((resolution, resolution), float(eps_slab))
    coords = np.linspace(-0.5, 0.5, resolution)
    xx, yy = np.meshgrid(coords, coords)
    eps[xx**2 + yy**2 < radius**2] = float(eps_hole)
    return eps


def slab_eps_map(
    slab_material: str,
    radius: float,
    hole_material: str = "air",
    resolution: int = 512,
) -> np.ndarray:
    """Convenience wrapper taking material *names* instead of permittivities."""
    return square_lattice_holes(
        eps_slab=eps_of(slab_material),
        radius=radius,
        eps_hole=eps_of(hole_material),
        resolution=resolution,
    )
