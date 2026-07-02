"""N_m (harmonic truncation) policy for dataset generation.

`generate_dataset.py` calls `n_m_for_twist` to choose N_m per design. The
mechanism is a theta -> N_m step function; v1 happens to use a single flat value
(see below), but the shape is kept so a future converged study can install a
real per-twist (or per-material) policy without code changes.

WHY v1 IS A FLAT N=2 (decided 2026-06-30)
-----------------------------------------
The convergence study (`data_generation/convergence_study.py`) measured two
things that overruled the original "adaptive-by-twist" plan:

1. COST WALL. On this hardware a twisted N=3 spectrum costs ~33 min (~220 s per
   wavelength); cost grows as (2N+1)^4. A converged (N>=3) dataset of any useful
   size is multi-day -- infeasible for now. N=2 (~1.5 min/spectrum at the study
   grid) is the practical ceiling.
2. NO CLEAN TWIST TREND. Across the sampled 8-30 deg range, Si3N4 convergence is
   NOT monotonic in twist -- the absolute CD error is confounded by CD magnitude
   (high-twist designs have larger CD, so they trip an absolute tolerance first).
   So binning N_m by twist is not supported by the data.

v1 therefore uses a flat, cost-driven N=2. This is accurate at low twist (N=2
within tol for Si3N4 at theta~8-12 deg) but UNDER-RESOLVED at high twist and for
higher-contrast materials (TiO2, a-Si), where the CD can be off by ~0.05-0.07.
Datasets built this way are "v0/approximate": fine for standing up the forward
surrogate, not for final device numbers. See the `solver-cost-and-convergence-v1`
project note and datasets/convergence/convergence_summary.json (git-ignored).

To install a real policy later, run the study at higher N (ideally overnight / on
faster hardware), then `load_policy_from_summary` and paste into N_M_POLICY.
"""

from __future__ import annotations

import json

# theta_min (deg) -> N_m, sorted ascending by theta_min with NON-INCREASING N_m.
# For a given twist, use the N_m of the largest theta_min <= twist.
# v1: a single flat bin (cost-driven N=2; twist-binning unsupported by v1 data).
N_M_POLICY: list[tuple[float, int]] = [
    (0.0, 2),
]


def n_m_for_twist(theta_deg: float, policy: list[tuple[float, int]] | None = None) -> int:
    """Harmonic truncation N_m to use for a design at twist `theta_deg`.

    Returns the N_m of the last bin whose lower edge is <= theta_deg. Below the
    smallest edge it falls back to the first (hardest) bin's N_m.
    """
    policy = policy if policy is not None else N_M_POLICY
    n_m = policy[0][1]
    for theta_min, value in policy:
        if theta_deg >= theta_min:
            n_m = value
        else:
            break
    return n_m


def load_policy_from_summary(summary_path: str) -> list[tuple[float, int]]:
    """Read recommended_policy out of a convergence_summary.json file.

    Convenience for refreshing N_M_POLICY after a study run; not used at dataset
    generation time (we read the version-controlled constant instead).
    """
    with open(summary_path) as f:
        summary = json.load(f)
    return [(r["theta_min"], int(r["N_m"])) for r in summary["recommended_policy"]]
