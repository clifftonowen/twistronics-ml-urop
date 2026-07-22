"""Fano-lineshape resonance fitting: turn a raw T(lambda) spectrum into a
handful of physically meaningful (lambda0, Q, amplitude) numbers instead of a
dense pointwise curve.

Motivation (see EXPERIMENTS.md Sec 6c): the 26-point/600-850nm campaign grid
was shown to alias the true CD spectrum -- sharp resonant features (1.7-12.5nm
wide) fall between sample points spaced ~10nm apart, so direct regression on
the raw grid has no chance of recovering the true structure-to-response map.
Fitting each polarization's transmission resonances separately -- rather than
the raw CD curve, which is a normalized difference of two resonances and not
itself a natural single-Fano quantity -- gives a smooth, low-dimensional target
that is the direct computational analogue of CLAUDE.md's own recipe for CD:
"a pair of chiral eigenmodes of opposite handedness, closely spaced in
frequency". Q is also half of the project's primary device metric (CD, Q) and
nothing upstream of this module measures it.

This module has no solver dependency -- it operates on any (lambda, y) array
pair, real or synthetic, which is what lets it be validated cheaply before any
new RCWA simulation is spent on a denser dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def fano_lineshape(lam, lam0, gamma, q, amp, bg):
    """Fano resonance in wavelength.

    eps = 2*(lam-lam0)/gamma; T = bg + amp*(q+eps)^2/(1+eps^2).
    q -> large recovers a symmetric Lorentzian peak; q ~ 0 gives a symmetric
    dip (antiresonance) -- both shapes were visible in the dense CD/T scans.
    """
    eps = 2.0 * (np.asarray(lam) - lam0) / gamma
    return bg + amp * (q + eps) ** 2 / (1.0 + eps ** 2)


@dataclass
class FitResult:
    lam0: float
    gamma: float
    q: float
    amp: float
    bg: float
    r2: float
    success: bool
    kind: str  # "peak" | "dip", from the candidate detector
    window_nm: float
    n_points: int

    @property
    def Q(self) -> float:
        """Standard photonics Q-factor: resonance wavelength / linewidth."""
        return self.lam0 / self.gamma if self.gamma > 0 else float("nan")


def _robust_noise_estimate(y: np.ndarray) -> float:
    """Estimate the point-to-point noise floor, ignoring smooth large-scale
    structure (a resonance's own amplitude would otherwise swamp a simple
    std(y)-based threshold -- see EXPERIMENTS.md Sec 6d for how this bit the
    first version of this module). Uses the MAD of first differences, scaled
    to be a consistent std estimator under Gaussian noise.
    """
    d = np.diff(y)
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / np.sqrt(2.0)  # /sqrt(2): diff of 2 samples doubles variance


def detect_candidates(lam: np.ndarray, y: np.ndarray, prominence: float | None = None):
    """Find candidate resonance centers as local peaks AND dips in y.

    Returns a list of {"lam0_guess": float, "index": int, "kind": "peak"|"dip"}.
    Default prominence is set from a robust point-to-point noise estimate (NOT
    a fraction of std(y) -- a strong resonance's own amplitude dominates std(y)
    and lets genuine noise in the flat background pass straight through).
    """
    y = np.asarray(y, dtype=float)
    if prominence is None:
        noise = _robust_noise_estimate(y)
        prominence = max(8.0 * noise, 1e-6)

    peak_idx, _ = find_peaks(y, prominence=prominence)
    dip_idx, _ = find_peaks(-y, prominence=prominence)

    candidates = [{"lam0_guess": float(lam[i]), "index": int(i), "kind": "peak"} for i in peak_idx]
    candidates += [{"lam0_guess": float(lam[i]), "index": int(i), "kind": "dip"} for i in dip_idx]
    candidates.sort(key=lambda c: c["lam0_guess"])
    return candidates


def fit_resonance(lam: np.ndarray, y: np.ndarray, lam0_guess: float,
                   window_nm: float = 60.0, kind: str = "peak",
                   min_gamma: float | None = None) -> FitResult:
    """Fit one Fano resonance in a window around lam0_guess.

    `min_gamma` floors the fitted linewidth. Without this, `curve_fit` can
    (and did, on a wide-frequency-grid rescan with 2.5-8nm point spacing --
    see EXPERIMENTS.md) drive gamma toward its lower bound and produce a
    needle-thin spike that best-fits a couple of nearby points while being
    physically meaningless everywhere else -- observed Q up to ~1.5e5, when
    Q>~1000 is already far beyond what N=3/this grid could resolve or claim.
    A resonance narrower than ~1-2 grid points is not identifiable from this
    data, so the fit must not be allowed to claim one. Default (None) derives
    a floor from the window's own local point spacing.
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.abs(lam - lam0_guess) <= window_nm / 2.0
    lam_w, y_w = lam[mask], y[mask]

    if lam_w.size < 6:
        return FitResult(lam0_guess, float("nan"), float("nan"), float("nan"),
                          float("nan"), 0.0, False, kind, window_nm, int(lam_w.size))

    if min_gamma is None:
        local_spacing = float(np.median(np.abs(np.diff(np.sort(lam_w)))))
        min_gamma = max(2.0 * local_spacing, 1e-3)

    # Initial guesses from the local window shape.
    edge_n = max(2, lam_w.size // 6)
    bg0 = float(np.mean(np.concatenate([y_w[:edge_n], y_w[-edge_n:]])))
    center_idx = int(np.argmin(np.abs(lam_w - lam0_guess)))
    amp0 = float(y_w[center_idx] - bg0)
    if amp0 == 0:
        amp0 = 1e-3 if kind == "peak" else -1e-3
    gamma0 = max(window_nm / 3.0, min_gamma * 1.5)
    q0 = 1.0 if kind == "peak" else 0.1

    # bg (baseline transmission) and gamma (linewidth) both need bounds tied
    # to the actual window/data, not generic fixed constants -- a fixed
    # bg in [-10, 10] and gamma up to window_nm*5 let curve_fit wander to
    # physically meaningless solutions (observed: bg=-8.2 when T is bounded
    # in [0,1], paired with gamma=212nm inside a 42.5nm window -- a
    # "resonance" 5x wider than the window it was fit in isn't identifiable
    # from that window; it's an unconstrained broad trend, not a resonance).
    y_lo, y_hi = float(np.min(y_w)), float(np.max(y_w))
    y_span = max(y_hi - y_lo, 1e-6)
    bg_lo, bg_hi = y_lo - 2.0 * y_span, y_hi + 2.0 * y_span

    p0 = [lam0_guess, gamma0, q0, amp0, bg0]
    lo = [lam_w.min(), min_gamma, -1e3, -10 * abs(amp0) - 1e-6, bg_lo]
    hi = [lam_w.max(), window_nm, 1e3, 10 * abs(amp0) + 1e-6, bg_hi]

    try:
        popt, _ = curve_fit(fano_lineshape, lam_w, y_w, p0=p0, bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError):
        return FitResult(lam0_guess, float("nan"), float("nan"), float("nan"),
                          float("nan"), 0.0, False, kind, window_nm, int(lam_w.size))

    y_fit = fano_lineshape(lam_w, *popt)
    ss_res = float(np.sum((y_w - y_fit) ** 2))
    ss_tot = float(np.sum((y_w - y_w.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    # Physical-plausibility gate: T_RCP/T_LCP are transmission coefficients,
    # strictly in [0,1]. A fit whose curve dips outside a small margin of
    # that range anywhere in the window is not capturing the physics, no
    # matter its aggregate r2 -- observed case: r2=0.80 (just above the 0.8
    # quality bar callers use) while the fitted curve went to T=-0.05 where
    # the true data was 0.53, because a single Fano compromised between a
    # real resonance on one side of the window and an unrelated rising edge/
    # shoulder on the other. Aggregate r2 alone doesn't catch this; a direct
    # physical-range check does.
    if np.min(y_fit) < -0.1 or np.max(y_fit) > 1.1:
        return FitResult(lam0_guess, float("nan"), float("nan"), float("nan"),
                          float("nan"), 0.0, False, kind, window_nm, int(lam_w.size))

    lam0_fit, gamma_fit, q_fit, amp_fit, bg_fit = (float(v) for v in popt)
    return FitResult(lam0_fit, gamma_fit, q_fit, amp_fit, bg_fit, r2, True, kind, window_nm, int(lam_w.size))


def fit_spectrum(lam: np.ndarray, y: np.ndarray, prominence: float | None = None,
                  window_nm: float = 60.0, dedup_frac_gamma: float = 0.3) -> list[FitResult]:
    """Detect candidate resonances in y(lam) and fit each one independently.

    An asymmetric Fano lineshape (finite q) genuinely has BOTH a local max and
    a local min close together (the zero of the Fano formula's numerator is a
    true minimum of the non-negative bracket term, distinct from its peak) --
    so `detect_candidates` legitimately finds 2 extrema per single physical
    resonance whenever q is not close to 0 or +-infinity. Fitting each extremum
    independently recovers the SAME underlying (lam0, gamma) from either side.

    Two fits are merged (as one resonance detected twice) only if their
    recovered lam0 differ by less than `dedup_frac_gamma` of the SMALLER
    fitted linewidth -- i.e. they've converged to essentially the same center.
    This is deliberately much tighter than a fixed nm distance: two genuinely
    distinct, closely-spaced resonances (e.g. the opposite-handedness mode
    pairs CD depends on, CLAUDE.md's "pair of chiral eigenmodes... closely
    spaced in frequency") have fitted centers separated by a sizeable fraction
    of their own linewidth and must NOT be merged, even if only a few nm apart
    in absolute terms.

    Returns fits sorted by lam0, including failed/low-r2 ones (deduplication
    only merges successful fits) -- callers decide what quality bar to apply.
    """
    candidates = detect_candidates(lam, y, prominence=prominence)

    # Local grid spacing near each candidate -- needed because a grid uniform
    # in FREQUENCY (not wavelength) has spacing that varies severalfold across
    # the span (observed: ~2.5nm near the short-wavelength/high-f end vs
    # ~8nm near the long-wavelength/low-f end of a 555-1000nm wide-f-grid
    # rescan). A fixed nm floor tuned for one end starves fits at the other:
    # a 4.0nm floor gave only 2-3 points (below the fit_resonance minimum of
    # 6) for genuine, narrow (~2-3nm-wide) resonances found at the fine-
    # spacing end, dropping yield from ~80-100% (narrow uniform-wavelength
    # grid) to ~30-50% (wide uniform-frequency grid) -- see EXPERIMENTS.md.
    lam_sorted = np.sort(np.asarray(lam))
    local_spacing = np.median(np.diff(lam_sorted))  # coarse global proxy is fine here

    # Cap each candidate's fitting window by its distance to the nearest OTHER
    # candidate. Two independent resonances closer together than window_nm
    # would otherwise both sit inside one fit's window, and a single-Fano
    # model fit to a two-resonance superposition is not meaningful -- this is
    # exactly the closely-spaced-mode-pair regime CD depends on (CLAUDE.md's
    # "pair of chiral eigenmodes... closely spaced in frequency"), so it must
    # be handled here rather than assumed away. The floor ensures at least
    # ~8 grid points regardless of local grid resolution.
    guesses = [c["lam0_guess"] for c in candidates]
    min_floor = max(8.0 * local_spacing, 4.0)
    fits = []
    for i, c in enumerate(candidates):
        others = [g for j, g in enumerate(guesses) if j != i]
        if others:
            nearest_gap = min(abs(c["lam0_guess"] - g) for g in others)
            local_window = min(window_nm, max(2.0 * nearest_gap, min_floor))
        else:
            local_window = window_nm
        fits.append(fit_resonance(lam, y, c["lam0_guess"], window_nm=local_window, kind=c["kind"]))

    fits.sort(key=lambda f: f.lam0 if f.success else float("inf"))
    merged: list[FitResult] = []
    for f in fits:
        if (f.success and merged and merged[-1].success
                and abs(f.lam0 - merged[-1].lam0) <= dedup_frac_gamma * min(f.gamma, merged[-1].gamma)):
            if f.r2 > merged[-1].r2:
                merged[-1] = f
        else:
            merged.append(f)
    return merged


@dataclass
class ModePair:
    rcp: FitResult
    lcp: FitResult
    predicted_peak_abs_cd: float = 0.0
    splitting_nm: float = field(init=False)

    def __post_init__(self):
        self.splitting_nm = abs(self.rcp.lam0 - self.lcp.lam0)


def _predicted_peak_abs_cd(rcp: FitResult, lcp: FitResult, n_eval: int = 200) -> float:
    """Peak |CD| implied by two FITTED Fano curves alone (no raw data needed).

    Evaluates both modes' fitted lineshapes and returns the max |CD| the pair
    predicts. Used to score candidate RCP/LCP pairings by actual chirality
    potential, not just splitting distance -- see EXPERIMENTS.md Sec 6d:
    distance-only nearest-neighbor matching missed the true CD-maximizing pair
    in a dense cluster (design 48), because a smaller splitting doesn't always
    mean stronger CD once resonance amplitude/Q differences are accounted for.

    Evaluation is restricted to the OVERLAP of each mode's own trusted fitting
    window (`FitResult.window_nm` around its `lam0`) -- a first version of
    this evaluated over a window scaled by `gamma`, which for a low-Q/broad
    resonance extrapolates the fitted curve far outside where it was actually
    constrained, producing unphysical predicted |CD| (observed: 27.7, when CD
    is bounded in [-1,1]). T is also clipped to [0, 1] as a physical guard
    against any residual extrapolation artifact.
    """
    lo = max(rcp.lam0 - rcp.window_nm / 2.0, lcp.lam0 - lcp.window_nm / 2.0)
    hi = min(rcp.lam0 + rcp.window_nm / 2.0, lcp.lam0 + lcp.window_nm / 2.0)
    if hi <= lo:
        # Windows don't overlap at all -- fall back to a small, conservative
        # span around the midpoint rather than extrapolating either fit.
        mid = 0.5 * (rcp.lam0 + lcp.lam0)
        half_span = 0.25 * min(rcp.gamma, lcp.gamma)
        lo, hi = mid - half_span, mid + half_span

    lam = np.linspace(lo, hi, n_eval)
    t_rcp = np.clip(fano_lineshape(lam, rcp.lam0, rcp.gamma, rcp.q, rcp.amp, rcp.bg), 0.0, 1.0)
    t_lcp = np.clip(fano_lineshape(lam, lcp.lam0, lcp.gamma, lcp.q, lcp.amp, lcp.bg), 0.0, 1.0)
    denom = t_rcp + t_lcp
    cd = np.divide(t_rcp - t_lcp, denom, out=np.zeros_like(denom), where=denom > 1e-6)
    return float(np.max(np.abs(cd)))


def match_rcp_lcp_modes(res_rcp: list[FitResult], res_lcp: list[FitResult],
                         max_split_nm: float = 15.0, min_r2: float = 0.9) -> list[ModePair]:
    """Optimal RCP/LCP mode pairing, weighted by predicted chirality.

    Default min_r2=0.9 (raised from an initial 0.5/0.8 -- see EXPERIMENTS.md):
    borderline r2 in [0.8, 0.9) fits were found to admit compromise fits (a
    single Fano straddling a real resonance and an unrelated rising edge, or
    a large-q/tiny-amp/large-gamma near-degenerate solution) that pass the
    physical-plausibility bounds in `fit_resonance` yet still distort
    downstream CD predictions (observed: a saturated predicted|CD|=1.000 from
    an r2=0.84 fit). 0.9 cleanly excludes these in practice while the
    genuinely well-fit resonances validated in EXPERIMENTS.md Sec 6d score
    r2>0.98-0.999.

    Only fits with r2 >= min_r2 are eligible, and a pair is only feasible if
    its splitting is within max_split_nm (CLAUDE.md's CD recipe requires
    opposite-handedness modes closely spaced in frequency). Among feasible
    pairs, solves for the assignment that MAXIMIZES total predicted peak|CD|
    (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`), instead
    of greedily claiming the smallest-splitting pairs first -- greedy-by-
    distance was found to miss the true CD-maximizing pair whenever >=2
    resonances per polarization cluster within max_split_nm of each other
    (see EXPERIMENTS.md Sec 6d).
    """
    from scipy.optimize import linear_sum_assignment

    rcp_ok = [r for r in res_rcp if r.success and r.r2 >= min_r2]
    lcp_ok = [r for r in res_lcp if r.success and r.r2 >= min_r2]
    if not rcp_ok or not lcp_ok:
        return []

    n, m = len(rcp_ok), len(lcp_ok)
    size = max(n, m)
    cost = np.zeros((size, size))  # 0 = "no benefit", used for padding/infeasible pairs
    scores: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(m):
            if abs(rcp_ok[i].lam0 - lcp_ok[j].lam0) <= max_split_nm:
                score = _predicted_peak_abs_cd(rcp_ok[i], lcp_ok[j])
                scores[(i, j)] = score
                cost[i, j] = -score  # maximize score == minimize -score

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_ind, col_ind):
        if (i, j) in scores:  # drop assignments that landed on padding/infeasible slots
            pairs.append(ModePair(rcp=rcp_ok[i], lcp=lcp_ok[j], predicted_peak_abs_cd=scores[(i, j)]))
    return pairs
