"""Adaptive N_m policy: pick the harmonic truncation per design from its twist.

The twisted bilayer's required truncation N_m grows as the twist angle shrinks
(smaller angle -> larger moire supercell -> more harmonics needed). Paying the
global worst-case N_m on every design wastes compute on the easy large-twist
designs, whose cost scales as (2*N_m+1)^4. Instead `generate_dataset.py` calls
`n_m_for_twist` to select N_m per design.

The bins below are the OUTPUT of the convergence study
(`data_generation/convergence_study.py`); see the recommended_policy field of
`datasets/convergence/convergence_summary.json`. They live here, in version
control, so dataset generation is reproducible even though the study artifacts
are git-ignored. To refresh them after re-running the study, read that JSON
(`load_policy_from_summary`) and paste the result into N_M_POLICY.
"""

from __future__ import annotations

import json

# theta_min (deg) -> N_m, sorted ascending by theta_min with NON-INCREASING N_m
# (smaller twist never needs fewer harmonics). For a given twist, use the N_m of
# the largest theta_min <= twist -- i.e. the harder (smaller-angle) edge of its bin.
#
# PROVISIONAL (literature expectation: alpha=10 deg, eps_r=4 converges by N_m~3;
# smaller angles need more). REPLACE with the study's recommended_policy once
# `convergence_study.py` finishes. The mechanism below is final; only these
# numbers are pending.
N_M_POLICY: list[tuple[float, int]] = [
    (0.0, 4),    # theta < 10 deg: small twist, large supercell -> more harmonics
    (10.0, 3),   # theta >= 10 deg: converges sooner
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
