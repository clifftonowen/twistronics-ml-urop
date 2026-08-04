"""Peak-centred narrow-window spectral representation.

Motivation (EXPERIMENTS.md Sec 6c): the broad 600-850nm / 26-point campaign grid
aliases the CD spectrum -- 41-58% of the true spectral variance falls between
sample points. Three campaigns have now failed to make CD learnable on that
representation while transmission, which is smooth on the same grid, learns
fine. The target itself was the problem, not the model or the design count.

This module implements the replacement representation: instead of a broad,
coarsely-sampled spectrum, each design is measured on a NARROW window CENTRED ON
ITS OWN CD PEAK and sampled finely enough to resolve the resonance. Two
properties matter:

  * Resolution. Measured on the 20 already-dense-scanned designs, subsampling a
    1.67nm scan to 5nm loses up to 35% of the true peak height; at 3.3nm the
    worst case is 8%. A ~1nm window grid resolves what a 10nm broad grid cannot.

  * Alignment. Peak wavelengths across those designs span 558-791nm
    (f = a/lambda in 0.632-0.896), and 7 of 20 designs carry four or more lobes
    of comparable height. In a FIXED window a resonance drifting in or out as
    the geometry changes looks like a discontinuity in the structure->response
    map; centring the window on the peak removes that by construction.

The window is defined RELATIVELY, on u = f/f_peak - 1, not as a fixed number of
nanometres. Resonance width scales with Q, not with an absolute nm figure, so a
relative window covers the same Q range at every peak wavelength while a
fixed-nm window does not. `resolvable_q_range` states that coverage explicitly
rather than leaving it to a comment.

No solver dependency: everything here operates on plain (freq, value) arrays, so
it can be validated on synthetic spectra before any GPU time is spent
(`scripts/validate_peak_window.py`).

Conventions: `freqs` are normalized frequency f = a/lambda (dimensionless);
wavelengths are nm. Every function labels which one it takes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

# Window defaults. See `resolvable_q_range` for what these imply about which
# resonances the window can actually measure -- the point count is set by the
# observed Q distribution, not chosen for roundness.
DEFAULT_HALF_WIDTH = 0.03   # +-3% in f; ~44nm total at a 725nm peak
DEFAULT_WINDOW_PTS = 41     # ~1.1nm spacing at a 725nm peak


@dataclass(frozen=True)
class PeakLocation:
    """Where a design's dominant CD feature sits, and how isolated it is."""

    f_peak: float          # normalized frequency of max |CD|
    cd_at_peak: float      # SIGNED CD there (handedness matters)
    peak_abs_cd: float     # |CD| at the peak; the screening quantity
    index: int             # index into the source grid
    grid_step_f: float     # source grid spacing -> the centring uncertainty
    n_competing_lobes: int  # lobes >= `lobe_frac` x peak, including the peak
    fwhm_u: float          # apparent relative FWHM on the SOURCE grid
    q_estimate: float      # 1/fwhm_u -- a lower bound on the true Q (see below)
    fwhm_grid_pts: float   # apparent FWHM in source grid steps

    @property
    def width_is_grid_limited(self) -> bool:
        """True when the peak spans too few source samples to have a measured
        width -- its reported `q_estimate` is then a LOWER BOUND, not a value.

        This matters before pass 2 runs: three of the twenty existing dense
        designs have peaks only 2-3 samples wide on a 0.004-in-f grid, implying
        Q >~ 220. Near a Fabry-Perot BIC the radiative Q scales as (gap
        deviation)^-2, so arbitrarily narrow resonances genuinely exist in this
        parameter space -- they are not measurement artefacts, and a window
        sized for typical resonances will under-resolve them.
        """
        return self.fwhm_grid_pts <= 3.0

    def lattice_constant_nm(self, lam_target_nm: float) -> float:
        """Lattice constant that places this peak at `lam_target_nm`.

        Valid only while materials are non-dispersive: the solver output then
        depends on f alone, so a_nm is a free post-hoc relabelling rather than a
        design dimension (README "Conventions"). This makes f_peak a
        FABRICATION OUTPUT -- the lattice constant to pattern -- which is why
        splitting it out of the spectrum is physically legitimate and not just
        a modelling convenience.
        """
        return self.f_peak * lam_target_nm


def _as_sorted_freq(freqs: np.ndarray, values: np.ndarray):
    """Return (freqs, values) sorted ascending in f.

    Grids built from a descending wavelength sweep arrive descending in f; every
    consumer here assumes ascending, so normalize once at the entry point.
    """
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(freqs)
    return freqs[order], values[order]


def find_competing_lobes(freqs: np.ndarray, cd: np.ndarray, lobe_frac: float = 0.5) -> np.ndarray:
    """Indices of |CD| lobes reaching at least `lobe_frac` x the global peak.

    The count is this project's label-swap risk metric. "The peak" is only a
    well-defined regression target if it is the clear winner; when several lobes
    are comparable, a small geometry change can promote a different one and the
    target jumps discontinuously. Measured on the existing dense scans, 7 of 20
    designs have four or more such lobes -- common enough that it has to be
    recorded per design and correlated against surrogate error, not assumed away.

    Operates on |CD| so that the two lobes flanking a Fano zero-crossing both
    count (they are genuinely competing extrema of the response).

    Endpoints are checked explicitly: `find_peaks` cannot return the first or
    last sample, but a peak sitting at the edge of the scanned band is exactly
    the case where the band was too narrow, so it must not be silently dropped.
    """
    freqs, cd = _as_sorted_freq(freqs, cd)
    a = np.abs(cd)
    if a.size < 3:
        return np.array([], dtype=int)
    peak = float(a.max())
    if peak <= 0:
        return np.array([], dtype=int)
    height = lobe_frac * peak

    idx, _ = find_peaks(a, height=height)
    idx = list(idx)
    if a[0] >= height and a[0] > a[1]:
        idx.append(0)
    if a[-1] >= height and a[-1] > a[-2]:
        idx.append(a.size - 1)
    return np.array(sorted(set(idx)), dtype=int)


def _apparent_fwhm(freqs: np.ndarray, cd: np.ndarray, i: int) -> tuple[float, float]:
    """Half-maximum width of the |CD| feature at index `i`, from the grid alone.

    Walks outward to the first sample below half the peak, then linearly
    interpolates the crossing. Returns (relative FWHM, FWHM in grid steps). If
    the feature runs off the end of the grid the available side is mirrored, so
    the result is a width, not a NaN -- an edge-truncated peak still needs a
    number for the caller to act on.
    """
    a = np.abs(cd)
    half = 0.5 * a[i]
    step = float(np.median(np.diff(freqs)))

    def cross(direction: int) -> float | None:
        """Frequency where |CD| falls to half, walking outward from the peak."""
        j = i
        while 0 <= j + direction < a.size:
            prev, j = j, j + direction
            if a[j] <= half:
                # Linear interpolation between `prev` (above half) and `j`
                # (below). `freqs[j] - freqs[prev]` already carries the sign of
                # the walk direction, so no extra sign factor belongs here.
                span = a[prev] - a[j]
                frac = (a[prev] - half) / span if span > 0 else 0.0
                return float(freqs[prev] + frac * (freqs[j] - freqs[prev]))
        return None

    lo, hi = cross(-1), cross(+1)
    if lo is None and hi is None:
        return float("nan"), float("nan")
    # A peak running off one end of the grid still needs a width: mirror the
    # side that was measurable rather than returning NaN.
    if lo is None:
        lo = freqs[i] - (hi - freqs[i])
    if hi is None:
        hi = freqs[i] + (freqs[i] - lo)

    # Floor the width at one grid step. A peak spanning a single sample has no
    # measurable width, and reporting the sub-step interpolated number would
    # claim a Q the data cannot support; one step makes `q_estimate` an honest
    # lower bound, which `width_is_grid_limited` then labels as such.
    fwhm_f = max(hi - lo, step)
    return fwhm_f / float(freqs[i]), fwhm_f / step


def locate_peak(freqs: np.ndarray, cd: np.ndarray, lobe_frac: float = 0.5) -> PeakLocation:
    """Find the dominant CD feature: global argmax |CD| on the given grid.

    Deliberately NOT sub-grid refined (e.g. by a parabolic vertex fit). The
    window this feeds is +-3% in f while a realistic locate grid step is ~0.004
    in f -- roughly a fifth of the window half-width -- so refinement cannot
    change whether the true peak lands inside the window. Worse, most peaks in
    the existing scans are only 2-3 samples wide on the locate grid, where a
    parabolic vertex would report false precision. `grid_step_f` is returned
    instead so callers can see the real centring uncertainty.
    """
    freqs, cd = _as_sorted_freq(freqs, cd)
    if freqs.size < 2:
        raise ValueError("need at least 2 grid points to locate a peak")
    i = int(np.argmax(np.abs(cd)))
    fwhm_u, fwhm_pts = _apparent_fwhm(freqs, cd, i)
    return PeakLocation(
        f_peak=float(freqs[i]),
        cd_at_peak=float(cd[i]),
        peak_abs_cd=float(abs(cd[i])),
        index=i,
        grid_step_f=float(np.median(np.diff(freqs))),
        n_competing_lobes=int(find_competing_lobes(freqs, cd, lobe_frac).size),
        fwhm_u=fwhm_u,
        q_estimate=1.0 / fwhm_u if fwhm_u > 0 else float("nan"),
        fwhm_grid_pts=fwhm_pts,
    )


def relative_grid(
    f_peak: float,
    half_width: float = DEFAULT_HALF_WIDTH,
    n_pts: int = DEFAULT_WINDOW_PTS,
) -> np.ndarray:
    """Normalized-frequency grid for the window: f_peak * (1 + u), u uniform.

    Uniform in f (hence in u) rather than in wavelength. Over a +-3% span the
    two differ by ~6%, which is immaterial, and uniform-in-u is what makes the
    axis common across designs -- the whole point of the representation.
    """
    if n_pts < 3:
        raise ValueError("n_pts must be >= 3")
    return float(f_peak) * (1.0 + relative_axis(half_width, n_pts))


def relative_axis(
    half_width: float = DEFAULT_HALF_WIDTH,
    n_pts: int = DEFAULT_WINDOW_PTS,
) -> np.ndarray:
    """The common axis u = f/f_peak - 1 shared by every design's window.

    This is the x-axis the surrogate's spectral target is defined on; f_peak is
    carried separately as a scalar output.
    """
    return np.linspace(-half_width, half_width, n_pts)


def relative_grid_wavelengths_nm(
    f_peak: float,
    a_nm: float,
    half_width: float = DEFAULT_HALF_WIDTH,
    n_pts: int = DEFAULT_WINDOW_PTS,
) -> np.ndarray:
    """The window as wavelengths (nm), which is what the solver call takes.

    Returned ascending in f, i.e. DESCENDING in wavelength, matching
    `relative_grid` element-for-element so results need no reordering.
    """
    return a_nm / relative_grid(f_peak, half_width, n_pts)


def resample_to_relative(
    freqs: np.ndarray,
    values: np.ndarray,
    f_peak: float,
    half_width: float = DEFAULT_HALF_WIDTH,
    n_pts: int = DEFAULT_WINDOW_PTS,
) -> np.ndarray:
    """Interpolate an existing spectrum onto this design's relative window.

    For re-processing already-simulated dense shards into the new
    representation. Points falling outside the source grid return NaN rather
    than an extrapolated value: `CubicSpline` extrapolates without complaint and
    doing so produced a nonsense result once already (EXPERIMENTS.md Sec 6e,
    where out-of-range extrapolation gave an RMS of ~10 on a quantity bounded in
    [-1, 1]). A NaN is a visible failure; a wild number is not.
    """
    freqs, values = _as_sorted_freq(freqs, values)
    target = relative_grid(f_peak, half_width, n_pts)
    out = np.full(target.shape, np.nan)
    inside = (target >= freqs[0]) & (target <= freqs[-1])
    if inside.any():
        out[inside] = CubicSpline(freqs, values)(target[inside])
    return out


@dataclass(frozen=True)
class QRange:
    """Which resonances a given window can measure.

    Q = f0 / FWHM_f = 1 / (relative FWHM), so on a relative axis the window's
    reach is a pure function of its half-width and point count.
    """

    q_min: float            # broadest resonance whose FWHM still fits in the window
    q_max_resolved: float   # narrowest with >= `pts_per_fwhm` samples across FWHM
    q_max_detectable: float  # narrowest still visible at all (2 samples across FWHM)
    step_u: float           # grid spacing in relative units

    def classify(self, q: float) -> str:
        """Label a fitted Q against this window: 'too_broad' | 'resolved' |
        'marginal' | 'unresolved'. Used by the per-design resolution self-check
        -- a Q pressed against the ceiling means the window is repeating, at a
        finer scale, exactly the aliasing this representation exists to fix.
        """
        if not np.isfinite(q) or q <= 0:
            return "unresolved"
        if q < self.q_min:
            return "too_broad"
        if q <= self.q_max_resolved:
            return "resolved"
        if q <= self.q_max_detectable:
            return "marginal"
        return "unresolved"


def peak_recovery(q: float, half_width: float = DEFAULT_HALF_WIDTH,
                  n_pts: int = DEFAULT_WINDOW_PTS, n_phases: int = 41) -> float:
    """Worst-case fraction of a resonance's peak height this window recovers.

    `QRange.classify` returns a conservative LABEL; this returns the number that
    actually matters. A resonance whose centre falls between two samples is
    measured short, and how short depends on the grid phase, which is arbitrary
    -- so this scans the phase and reports the worst case.

    Assumes a Lorentzian, which understates recovery slightly for a Fano of the
    same linewidth (its peak is broader than its half-width suggests).

    Context for reading the result: at the +-3% x 41-point default, a Q=172
    resonance (the median measured on the first eight designs) recovers 0.94,
    against 0.42 on the 5nm grid this representation replaced. "Marginal" by the
    classifier is still a large practical improvement.
    """
    qr = resolvable_q_range(half_width, n_pts)
    f0 = 1.0
    hwhm = 0.5 * f0 / q
    worst = 1.0
    for shift in np.linspace(-0.5, 0.5, n_phases):
        grid = relative_grid(f0, half_width, n_pts) + shift * qr.step_u * f0
        worst = min(worst, float((1.0 / (1.0 + ((grid - f0) / hwhm) ** 2)).max()))
    return worst


def resolvable_q_range(
    half_width: float = DEFAULT_HALF_WIDTH,
    n_pts: int = DEFAULT_WINDOW_PTS,
    pts_per_fwhm: float = 5.0,
) -> QRange:
    """Q coverage of a window, computed rather than asserted.

    Lower bound: the FWHM must fit inside the window, so relative FWHM <=
    2*half_width and Q >= 1/(2*half_width).
    Upper bounds: a resonance needs `pts_per_fwhm` samples across its FWHM to be
    characterized (and a bare 2 to be seen at all), so relative FWHM >=
    pts_per_fwhm * step_u.

    Reference point for why the default point count is 41 and not 31: the
    largest Q fitted across the 20 existing dense designs was ~109
    (EXPERIMENTS.md Sec 6e) -- and that value was itself pressed against a
    grid-imposed floor on the fitted linewidth, so the true distribution very
    likely reaches higher. A 31-point window tops out at Q=100 resolved, i.e.
    below the already-observed maximum.

    These bounds describe REPRESENTATION -- whether the sampled points capture
    the lineshape. Fitting a Fano to it is a stricter test at the broad end,
    because a fit needs the wings and not just the FWHM. Measured against
    `resonances.fit_spectrum` on the default +-3% x 41-point window
    (synthetic ground truth, no noise): Q is recovered exactly (0.0% error,
    r2 = 1.0000) from Q = 40 up to at least 250, but Q <= 25 yields NO fit at
    all -- at Q = 25 the FWHM is 27nm inside a 40.6nm window, leaving no wings
    to constrain. So read q_min as the representation floor and ~40 as the
    practical Fano-fitting floor on this window.
    """
    step_u = 2.0 * half_width / (n_pts - 1)
    return QRange(
        q_min=1.0 / (2.0 * half_width),
        q_max_resolved=1.0 / (pts_per_fwhm * step_u),
        q_max_detectable=1.0 / (2.0 * step_u),
        step_u=step_u,
    )
