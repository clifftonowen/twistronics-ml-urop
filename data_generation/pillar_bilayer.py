"""Twisted bilayer of pillar arrays in a uniform medium (He et al. 2025 geometry).

The solver's gap layer is hard-coded to vacuum, so the uniform surrounding index
is handled by scaling: for non-magnetic media, dividing every permittivity by
n_bg^2 and multiplying frequency by n_bg leaves Maxwell's equations unchanged.

Uses the same 4D plane-wave basis and formulas as rcwa4d.solve_Smat_4D, arranged
to avoid forming the full scattering matrix:
  * pillar slices of one layer are combined while the matrices are still small
    (2NM x 2NM, one per spectator order), then coupled once to the other layer;
  * both half spaces and the gap are the background, so they contribute only
    propagation phases;
  * only the normal-incidence columns are solved for.
Equality with the stock solver is asserted in scripts/check_pillars.py.

Lengths are in units of the period. Light enters on the layer-1 side, and slices
are listed in the order light meets them (narrow cap first).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.special import j1

S2 = 1 / np.sqrt(2)
POL_RCP, POL_LCP = (S2, -1j * S2), (S2, 1j * S2)   # (pte, ptm), as simulate_spectra
_POOL = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) // 2)))

# Pillar of He et al. Fig. 5a: (height, radius) in nm; the cap closes at 230 nm.
PAPER_PROFILE = ((0, 100), (150, 90), (170, 87), (180, 84), (225, 27), (230, 0))


@dataclass
class PillarResult:
    wavelengths_nm: np.ndarray
    T0: np.ndarray      # (n, 2) zeroth-order transmission, columns RCP, LCP
    T: np.ndarray       # (n, 2) total transmission
    R: np.ndarray       # (n, 2) total reflection

    @staticmethod
    def _cd(t):
        s = t.sum(1)
        return np.divide(t[:, 0] - t[:, 1], s, out=np.zeros(len(s)), where=s > 1e-12)

    @property
    def cd0(self):
        return self._cd(self.T0)

    @property
    def cd(self):
        return self._cd(self.T)

    @property
    def residual(self):
        return np.abs(1 - self.T - self.R).max(1)


def profile_slices(profile=PAPER_PROFILE, n_cap=4, n_body=2):
    """(thickness, radius) in nm, cap first; body is the first profile segment."""
    h, r = np.array(profile, float).T
    edges = np.concatenate([np.linspace(0, h[1], n_body + 1),
                            np.linspace(h[1], h[-1], n_cap + 1)[1:]])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        z = np.linspace(lo, hi, 41)
        out.append((hi - lo, float(np.sqrt(np.mean(np.interp(z, h, r) ** 2)))))  # keeps volume
    return out[::-1]


def disc_conv(eps, radius, N):
    """Analytic convolution matrix of one disc (permittivity eps, background 1)."""
    g = np.arange(-N, N + 1)
    gx, gy = (a.ravel() for a in np.meshgrid(g, g))
    G = np.hypot(gx[:, None] - gx[None], gy[:, None] - gy[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (eps - 1) * radius * j1(2 * np.pi * G * radius) / G
    F[G == 0] = 1 + (eps - 1) * np.pi * radius ** 2
    return F.astype(complex)


def _block(a, b, c, d):
    return np.concatenate([np.concatenate([a, b], -1), np.concatenate([c, d], -1)], -2)


def _star(a, b):
    A11, A12, A21, A22 = a
    B11, B12, B21, B22 = b
    eye = np.eye(len(A11))
    D, F = eye - B11 @ A22, eye - A22 @ B11
    return (A11 + A12 @ np.linalg.solve(D, B11 @ A21), A12 @ np.linalg.solve(D, B12),
            B21 @ np.linalg.solve(F, A21), B22 + B21 @ np.linalg.solve(F, A22 @ B12))


def _kz(kx, ky):
    kz = np.sqrt((1 - kx ** 2 - ky ** 2).astype(complex))
    return np.where(kz.imag > 0, -kz, kz)      # solver convention: decaying = -i|kz|


def layer_smat(kx, ky, slices, k0):
    """S-matrix of a slice stack for one spectator order. slices: [(conv, inv_conv, thickness)]."""
    eye = np.eye(len(kx))
    kz = _kz(kx, ky)
    Kx, Ky = kx[:, None], ky[:, None]
    Kxx, Kxy, Kyy = np.diag(kx * kx), np.diag(kx * ky), np.diag(ky * ky)
    Vg = _block(Kxy, eye - Kxx, Kyy - eye, -Kxy) / (1j * np.concatenate([kz, kz]))
    tot = None
    for E, Ei, L in slices:
        P = _block(Kx * Ei * ky, eye - Kx * Ei * kx, Ky * Ei * ky - eye, -Ky * Ei * kx)
        Q = _block(Kxy, E - Kxx, Kyy - E, -Kxy)
        l2, W = np.linalg.eig(P @ Q)
        lam = np.sqrt(l2.astype(complex))
        V = Q @ W / lam
        a, b = np.linalg.inv(W), np.linalg.solve(V, Vg)
        A, B = a + b, a - b
        x = np.exp(-lam * L * k0)
        XB = x[:, None] * B
        AiX = np.linalg.solve(A, np.diag(x))
        M = np.linalg.inv(A - XB @ AiX @ B)
        S11 = M @ (XB @ AiX @ A - B)
        S12 = M @ (x[:, None] * (A - B @ np.linalg.solve(A, B)))
        s = (S11, S12, S12, S11)
        tot = s if tot is None else _star(tot, s)
    return tot


def _big_solve(F, rhs, device):
    if device == "gpu":
        import cupy as cp
        x = cp.linalg.solve(cp.asarray(F, cp.complex64), cp.asarray(rhs, cp.complex64))
        x = cp.asnumpy(x).astype(complex)
        cp.get_default_memory_pool().free_all_blocks()
        return x
    return np.linalg.solve(F, rhs)


def _solve_point(freq, twist, N, slices, gap, device):
    """Normal incidence, both circular polarisations -> (T0, T, R), each (2,)."""
    NM = (2 * N + 1) ** 2
    if abs(freq - 1) < 1e-6 or abs(freq - np.sqrt(2)) < 1e-6:
        freq += 2e-6    # exactly on a diffraction edge kz = 0 and the basis is singular
    k0 = 2 * np.pi * freq
    g = np.arange(-N, N + 1)
    gx, gy = (a.ravel().astype(float) for a in np.meshgrid(g, g))   # x fastest
    c, s = np.cos(twist), np.sin(twist)
    hx, hy = gx * c + gy * s, gy * c - gx * s                       # layer-2 lattice
    mid = NM // 2
    pols = np.array([[ptm, pte] for pte, ptm in (POL_RCP, POL_LCP)]).T  # rows x, y (TM=x, TE=y)

    if twist == 0:      # ordinary 2D problem; the 4D basis would be redundant
        kx, ky = -gx / freq, -gy / freq
        S = layer_smat(kx, ky, slices, k0)
        kz = _kz(kx, ky)
        inc = np.zeros((2, NM, 2), complex)
        inc[:, mid, :] = pols
        inc = inc.reshape(2 * NM, 2)
        X = np.tile(np.exp(-1j * kz * k0 * gap), 2)[:, None]
        A11, A12, A21, A22 = S
        B11, _, B21, _ = S
        v = np.linalg.solve(np.eye(2 * NM) - X * (A22 @ (X * B11)), X * (A21 @ inc))
        t = B21 @ v
        r = A11 @ inc + A12 @ (X * (B11 @ v))
    else:               # arrays below are indexed [g1, g2]
        kx = -(gx[:, None] + hx[None]) / freq
        ky = -(gy[:, None] + hy[None]) / freq
        kz = _kz(kx, ky)
        L1 = list(_POOL.map(lambda j: layer_smat(kx[:, j], ky[:, j], slices, k0), range(NM)))
        L2 = list(_POOL.map(lambda i: layer_smat(kx[i], ky[i], slices, k0), range(NM)))
        T1 = [np.stack([x[q] for x in L1]).reshape(NM, 2, NM, 2, NM) for q in range(4)]  # [g2,p,g1,q,h1]
        T2 = [np.stack([x[q] for x in L2]).reshape(NM, 2, NM, 2, NM) for q in range(4)]  # [g1,p,g2,q,h2]
        Xg = np.exp(-1j * kz * k0 * gap)
        X = Xg[None, :, :, None]
        inc = np.zeros((2, NM, NM, 2), complex)
        inc[:, mid, mid, :] = pols

        def a1(M, x):
            return np.einsum("jpiqh,qhjr->pijr", M, x, optimize=True)

        def a2(M, x):
            return np.einsum("ipjqh,qihr->pijr", M, x, optimize=True)

        # F = I - X A22 X B11; A22 is diagonal in g2, B11 diagonal in g1.
        AX = T1[3] * Xg.T[:, None, None, None, :]
        F = np.einsum("jpiqh,hqjsk->pijshk", AX, T2[0], optimize=True)
        F *= Xg[None, :, :, None, None, None]
        D = 2 * NM * NM
        F = F.reshape(D, D)
        np.negative(F, out=F)
        F[np.arange(D), np.arange(D)] += 1
        v = _big_solve(F, (X * a1(T1[2], inc)).reshape(D, 2), device).reshape(2, NM, NM, 2)
        t = a2(T2[2], v).reshape(D, 2)
        r = (a1(T1[0], inc) + a1(T1[1], X * a2(T2[0], v))).reshape(D, 2)
        kx, ky, kz = kx.ravel(), ky.ravel(), kz.ravel()

    def power(e):
        ex, ey = e.reshape(2, -1, 2)
        ez = (kx[:, None] * ex + ky[:, None] * ey) / kz[:, None]
        return (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2) * kz.real[:, None]

    pt, pr = power(t), power(r)
    return pt[len(pt) // 2], pt.sum(0), pr.sum(0)


def simulate_pillars(wavelengths_nm, n_tio2=2.4, gap_nm=120.0, twist_deg=22.6199, period_nm=400.0,
                     profile=PAPER_PROFILE, n_bg=1.46, N=2, device="gpu", n_cap=4, n_body=2):
    """CD spectrum of two identical pillar arrays separated by gap_nm (base to cap)."""
    eps = (n_tio2 / n_bg) ** 2
    sl = []
    for L, r in profile_slices(profile, n_cap, n_body):
        E = disc_conv(eps, r / period_nm, N)
        sl.append((E, np.linalg.inv(E), L / period_nm))
    lam = np.asarray(wavelengths_nm, float)
    out = [_solve_point(n_bg * period_nm / w, np.deg2rad(twist_deg), N, sl, gap_nm / period_nm, device)
           for w in lam]
    T0, T, R = (np.array(x) for x in zip(*out))
    return PillarResult(lam, T0, T, R)
