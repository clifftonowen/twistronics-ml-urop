"""Drive the RCWA-4D solver for one design over a wavelength grid -> CD spectrum.

This is the bridge between structural parameters X and the optical response Y.
For each wavelength we solve the twisted-bilayer scattering problem once and
read total transmission for right- and left-circular incidence, then form the
transmission circular dichroism

    CD(lambda) = (T_RCP - T_LCP) / (T_RCP + T_LCP)   in [-1, +1]

(CLAUDE.md section 3; transmission definition, fixed project-wide).

Layer stack handed to the solver (matches example3-convergence-check.ipynb):
    epsr_list   = [slab_eps, None, slab_eps]   # bottom slab, gap, top slab
    thickness   = [t,        d,    t]           # in units of a
    orientation = [1,        1,    2]           # layer 3 is the twisted copy
    gap_layer_indices = [1]                     # middle layer is vacuum gap

CONVENTIONS / KNOBS
  * The solver runs with a = 1 (normalized). Physical wavelength enters only as
    the normalized frequency  f~ = a_nm / lambda_nm  passed to set_freq_k.
  * Circular polarization is built from the solver's (pte, ptm) = (TE, TM)
    amplitudes. At normal incidence TE = y_hat, TM = x_hat, so
        RCP: (pte, ptm) = (1, -1j)/sqrt(2)
        LCP: (pte, ptm) = (1, +1j)/sqrt(2)
    The RCP/LCP *labelling* is a sign convention tied to the solver's exp(-iwt)
    time dependence; flipping it flips the sign of CD. What matters physically
    is consistency, and that twisting turns CD on while an untwisted C4v bilayer
    gives CD ~ 0 (symmetry sanity check, CLAUDE.md section 8).
  * For a fixed (freq, angle) the scattering matrix Sg depends only on geometry
    and k, not on the input polarization. set_freq_k() resets Sg; the first
    get_RT() solves it and the second reuses it -- so RCP and LCP cost ~one
    solve, not two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import data_generation  # noqa: F401  -> ensures rcwa4d is on sys.path
from rcwa4d import rcwa

from .geometry import slab_eps_map
from .parameter_sampler import DesignParams

_INV_SQRT2 = 1.0 / np.sqrt(2.0)
# (pte, ptm) amplitudes for circular incidence; see module docstring.
RCP = (_INV_SQRT2, -1j * _INV_SQRT2)
LCP = (_INV_SQRT2, +1j * _INV_SQRT2)


@dataclass
class SpectrumResult:
    """Optical response of one design over the wavelength grid."""

    wavelengths_nm: np.ndarray  # (n_lambda,)
    freqs: np.ndarray           # (n_lambda,) normalized f~ = a/lambda
    T_RCP: np.ndarray           # (n_lambda,) total transmission, RCP incidence
    T_LCP: np.ndarray           # (n_lambda,)
    R_RCP: np.ndarray           # (n_lambda,) total reflection, RCP (for energy check)
    R_LCP: np.ndarray           # (n_lambda,)
    cd: np.ndarray              # (n_lambda,) circular dichroism in [-1, 1]

    def energy_residual(self) -> np.ndarray:
        """|1 - (R + T)| per wavelength; ~0 for lossless real-epsilon media.

        A nonzero residual flags under-convergence (too small N_m) or a bug.
        """
        return np.abs(1.0 - (self.R_RCP + self.T_RCP))


def freqs_from_wavelengths(wavelengths_nm: np.ndarray, a_nm: float) -> np.ndarray:
    """Normalized frequency f~ = a/lambda used by the solver."""
    return a_nm / np.asarray(wavelengths_nm, dtype=float)


def simulate_design(
    params: DesignParams,
    wavelengths_nm: np.ndarray,
    a_nm: float = 500.0,
    N: int = 3,
    slab_material: str = "Si3N4",
    hole_material: str = "air",
    theta_inc_deg: float = 0.0,
    phi_inc_deg: float = 0.0,
    resolution: int = 512,
) -> SpectrumResult:
    """Compute the CD spectrum of one twisted-bilayer design.

    Args:
        params: the structure (twist, thickness, gap, radius; lengths in a).
        wavelengths_nm: wavelength grid to evaluate (e.g. 600..850 nm).
        a_nm: lattice constant in nm -- sets where features land in wavelength.
        N: harmonic truncation N_m (kept symmetric, M = N). Cost scales like
            (2N+1)^4 for the twisted bilayer; ALWAYS justify N with a
            convergence check (see convergence_sweep) before trusting a batch.
        slab_material, hole_material: see materials.py.
        theta_inc_deg, phi_inc_deg: incident direction (0,0 = normal).
        resolution: unit-cell grid resolution for the epsilon map.

    Returns:
        SpectrumResult with T/R for both handednesses and CD.
    """
    eps = slab_eps_map(slab_material, params.radius, hole_material, resolution)

    twist_rad = np.deg2rad(params.theta_deg)
    theta_rad = np.deg2rad(theta_inc_deg)
    phi_rad = np.deg2rad(phi_inc_deg)

    # One solver object reused across all wavelengths for this design.
    obj = rcwa(
        epsr_list=[eps, None, eps],
        thickness_list=[params.thickness, params.gap, params.thickness],
        orientation_list=[1, 1, 2],
        gap_layer_indices=[1],
        twist=twist_rad,
        N=N,
        M=N,
        a=1.0,
        verbose=0,
    )

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    freqs = freqs_from_wavelengths(wavelengths_nm, a_nm)

    n = len(freqs)
    T_RCP = np.empty(n)
    T_LCP = np.empty(n)
    R_RCP = np.empty(n)
    R_LCP = np.empty(n)

    for i, freq in enumerate(freqs):
        obj.set_freq_k(freq, (theta_rad, phi_rad))  # resets Sg
        (r_rcp, t_rcp), _ = obj.get_RT(*RCP)         # first call solves Sg
        (r_lcp, t_lcp), _ = obj.get_RT(*LCP)         # reuses cached Sg
        T_RCP[i], R_RCP[i] = float(t_rcp), float(r_rcp)
        T_LCP[i], R_LCP[i] = float(t_lcp), float(r_lcp)

    denom = T_RCP + T_LCP
    cd = np.divide(
        T_RCP - T_LCP, denom, out=np.zeros_like(denom), where=denom > 1e-12
    )

    return SpectrumResult(
        wavelengths_nm=wavelengths_nm,
        freqs=freqs,
        T_RCP=T_RCP,
        T_LCP=T_LCP,
        R_RCP=R_RCP,
        R_LCP=R_LCP,
        cd=cd,
    )


def convergence_sweep(
    params: DesignParams,
    wavelengths_nm: np.ndarray,
    N_values: list[int],
    **kwargs,
) -> dict[int, SpectrumResult]:
    """Run one design at several N_m values to judge convergence.

    Compare the returned spectra (e.g. max |CD(N) - CD(N_max)|) to pick the
    smallest N that is converged for the parameter regime you intend to sample
    (CLAUDE.md section 4: smaller twist / higher index contrast need larger N).
    """
    return {
        N: simulate_design(params, wavelengths_nm, N=N, **kwargs)
        for N in N_values
    }
