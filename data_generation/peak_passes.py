"""The two solver passes of the peak-centred campaign.

Split out from `peak_window.py` on purpose: that module is deliberately
solver-free so its maths can be validated on synthetic spectra at zero GPU cost.
This module is the thin layer that actually calls RCWA, and it is the only place
that knows both halves.

    pass 1  locate_pass  -- cheap, coarse-N, wide band -> where is the CD peak?
    pass 2  window_pass  -- expensive, converged-N, narrow window -> the target

Why two passes rather than one. Measured on the existing dense scans, the old
26-point/600-850nm grid finds the true peak within 5nm for only 11 of 20
designs and is off by 74-179nm for 7 of them -- it centres on the wrong lobe a
third of the time. So the locate pass must be genuinely dense (~0.004 in f),
which is affordable only at a lower truncation: on this hardware N=2 costs
~2 s/wavelength against ~34 s/wavelength at N=3, roughly 17x cheaper. Pass 2
then spends the converged-N budget only where it matters, on a window ~1/10th
the width of the old band.

The locate pass is only ever asked WHERE the peak is, never how tall it is, so
its lower truncation does not enter the training labels -- but that shifted
resonance positions are close enough between N=2 and N=3 is an assumption, not
a given, and `scripts/locate_gate.py` measures it before any campaign runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameter_sampler import DesignParams
from .peak_window import (
    DEFAULT_HALF_WIDTH,
    DEFAULT_WINDOW_PTS,
    PeakLocation,
    locate_peak,
    relative_axis,
    relative_grid_wavelengths_nm,
)
from .simulate_spectra import SpectrumResult, simulate_design

# Locate band in normalized frequency. Covers the 0.632-0.896 span of peaks
# observed across the 20 existing dense designs with margin on both sides.
# The upper limit stays below f = 1: at normal incidence that is the Rayleigh
# cutoff (lambda = a) where the first diffraction orders start to propagate,
# which both changes the physics and raises per-solve cost (EXPERIMENTS.md
# Sec 6 saw exactly this on a smoke shard swept to f = 1.10).
DEFAULT_LOCATE_F_MIN = 0.60
DEFAULT_LOCATE_F_MAX = 0.92
DEFAULT_LOCATE_PTS = 81      # step 0.004 in f, ~2.5nm at the band's blue end
DEFAULT_LOCATE_N = 2
DEFAULT_WINDOW_N = 3


@dataclass
class LocateResult:
    location: PeakLocation
    spectrum: SpectrumResult


@dataclass
class WindowResult:
    """Pass-2 output: one design's target, on the axis shared by all designs."""

    u: np.ndarray               # relative axis u = f/f_peak - 1
    f_peak: float               # carried separately; a fabrication output
    spectrum: SpectrumResult    # full solver output on the window grid
    half_width: float
    n_pts: int

    @property
    def cd(self) -> np.ndarray:
        return self.spectrum.cd

    @property
    def peak_abs_cd(self) -> float:
        return float(np.abs(self.spectrum.cd).max())


def locate_pass(
    params: DesignParams,
    a_nm: float = 500.0,
    f_min: float = DEFAULT_LOCATE_F_MIN,
    f_max: float = DEFAULT_LOCATE_F_MAX,
    n_pts: int = DEFAULT_LOCATE_PTS,
    N: int = DEFAULT_LOCATE_N,
    **sim_kwargs,
) -> LocateResult:
    """Scan the wide band coarsely to find where this design's CD peak is."""
    freqs = np.linspace(f_min, f_max, n_pts)
    res = simulate_design(params, a_nm / freqs, a_nm=a_nm, N=N, **sim_kwargs)
    return LocateResult(location=locate_peak(res.freqs, res.cd), spectrum=res)


def window_pass(
    params: DesignParams,
    f_peak: float,
    a_nm: float = 500.0,
    half_width: float = DEFAULT_HALF_WIDTH,
    n_pts: int = DEFAULT_WINDOW_PTS,
    N: int = DEFAULT_WINDOW_N,
    **sim_kwargs,
) -> WindowResult:
    """Resolve the resonance: converged-N scan of the peak-centred window."""
    lam = relative_grid_wavelengths_nm(f_peak, a_nm, half_width, n_pts)
    res = simulate_design(params, lam, a_nm=a_nm, N=N, **sim_kwargs)
    return WindowResult(
        u=relative_axis(half_width, n_pts),
        f_peak=float(f_peak),
        spectrum=res,
        half_width=half_width,
        n_pts=n_pts,
    )
