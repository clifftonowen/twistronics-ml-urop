"""Material relative permittivities (epsilon_r = n^2).

v1 uses NON-dispersive (wavelength-independent) constants. This is an explicit
approximation: across 600-850 nm the real materials below disperse by a few
percent, which shifts resonance wavelengths slightly. The RCWA solver takes a
single epsilon grid per solve, so to model true dispersion we would rebuild the
epsilon map per wavelength using n(lambda) from refractiveindex.info. That hook
is left for v2; for the first dataset we hold epsilon
fixed and document it in the dataset metadata.

Values are representative figures in the visible/NIR (~700 nm), chosen to match
the ranges quoted.
"""

from __future__ import annotations

# Relative permittivity (epsilon_r). n = sqrt(epsilon_r).
EPS = {
    "air": 1.0,        # holes / ambient
    "SiO2": 2.10,      # n ~ 1.45, low-index spacer  
    "Si3N4": 4.00,     # n ~ 2.00, high-index slab   
    "TiO2": 6.25,      # n ~ 2.50, high-index slab
    "aSi": 12.25,      # n ~ 3.50, amorphous Si (high contrast; needs larger N_m)
}

# Default high-index slab material for v1.
DEFAULT_SLAB = "Si3N4"


def eps_of(material: str) -> float:
    """Return epsilon_r for a named material, with a clear error otherwise."""
    try:
        return EPS[material]
    except KeyError as exc:
        raise KeyError(
            f"Unknown material {material!r}. Known: {sorted(EPS)}"
        ) from exc
