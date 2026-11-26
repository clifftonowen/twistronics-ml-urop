"""Checks for data_generation.pillar_bilayer. Exits nonzero on failure.

Usage: python -m scripts.check_pillars
"""
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

import data_generation  # noqa: F401
from data_generation import pillar_bilayer as pb
from rcwa4d import rcwa

bad = 0
TW = 22.6199


def check(name, ok, detail):
    global bad
    bad += not ok
    print(("PASS " if ok else "FAIL ") + name + "  " + detail, flush=True)


def stock(lam, N, gap_nm, n=2.4, theta=0.0, phi=0.0):
    """Same structure through the unmodified solver."""
    sl = pb.profile_slices()
    eps = (n / 1.46) ** 2
    dummy = np.ones((32, 32))
    k = len(sl)
    o = rcwa([dummy] * k + [None] + [dummy] * k,
             [s[0] / 400 for s in sl] + [gap_nm / 400] + [s[0] / 400 for s in sl],
             [1] * k + [1] + [2] * k, gap_layer_indices=[k],
             twist=np.deg2rad(TW), N=N, M=N, verbose=0)
    conv = [pb.disc_conv(eps, s[1] / 400, N) for s in sl]
    o.ER = conv + [None] + conv
    o.set_freq_k(1.46 * 400 / lam, (np.deg2rad(theta), np.deg2rad(phi)))
    T, T0 = [], []
    for p in (pb.POL_RCP, pb.POL_LCP):
        (_, t), (_, e) = o.get_RT(*p)
        T.append(float(np.real(t)))
        T0.append(float((np.abs(e) ** 2).sum()))
    return np.array(T0), np.array(T)


for N, lams in ((1, (560.0, 605.0, 640.0)), (2, (605.0,))):
    for lam in lams:
        a0, at = stock(lam, N, 120)
        b = pb.simulate_pillars([lam], N=N, device="cpu")
        d = max(np.abs(a0 - b.T0[0]).max(), np.abs(at - b.T[0]).max())
        check(f"stock match N={N} {lam:.0f}nm", d < 1e-5, f"max diff {d:.1e}")
for N, (th, ph) in ((1, (5.0, 30.0)), (1, (12.0, 0.0)), (2, (5.0, 30.0))):
    a0, at = stock(605.0, N, 120, theta=th, phi=ph)
    b = pb.simulate_pillars([605.0], N=N, device="cpu", theta_deg=th, phi_deg=ph)
    d = max(np.abs(a0 - b.T0[0]).max(), np.abs(at - b.T[0]).max())
    check(f"stock match oblique N={N} theta {th} phi {ph}", d < 1e-5, f"max diff {d:.1e}  T0 {b.T0[0].round(4)}")


# Index scaling: at p = 400 nm and 605 nm, solving with real eps and k0 in a n=1.46
# medium must equal the scaled problem. Independent route: one uniform slab of n=2.4,
# compared with the textbook Fabry-Perot formula evaluated in real units.
L, lam = 230.0, 605.0
eps = (2.4 / 1.46) ** 2
E = eps * np.eye(9, dtype=complex)
g = np.arange(-1, 2)
gx, gy = (q.ravel().astype(float) for q in np.meshgrid(g, g))
f = 1.46 * 400 / lam
S = pb.layer_smat(-gx / f, -gy / f, [(E, np.linalg.inv(E), L / 400)], 2 * np.pi * f)
t1 = abs(S[2][4, 4]) ** 2
r12 = (1.46 - 2.4) / (1.46 + 2.4)
ph = 2 * np.pi * 2.4 * L / lam
t2 = abs((1 - r12 ** 2) * np.exp(1j * ph) / (1 - r12 ** 2 * np.exp(2j * ph))) ** 2
check("index scaling vs slab formula", abs(t1 - t2) < 1e-6, f"{t1:.6f} vs {t2:.6f}")

# Oblique incidence on a uniform slab, TE and TM, against the Fresnel formula.
th = np.deg2rad(20.0)
for name, pol in (("TE", (1, 0)), ("TM", (0, 1))):
    E = eps * np.eye(9, dtype=complex)
    f = 1.46 * 400 / lam
    kx = np.sin(th) - gx / f
    S = pb.layer_smat(kx, -gy / f, [(E, np.linalg.inv(E), L / 400)], 2 * np.pi * f)
    te, tm = np.array([0, 1.0]), np.array([np.cos(th), 0.0])     # (Ex, Ey) of unit TE / TM fields
    inc = np.zeros(18, complex)
    v = te if name == "TE" else tm
    inc[4], inc[13] = v
    t = S[2] @ inc
    tx, ty = t[4], t[13]
    tz = -(np.sin(th) * tx) / np.cos(th)
    t1 = abs(tx) ** 2 + abs(ty) ** 2 + abs(tz) ** 2
    c1, n2 = np.cos(th), 2.4 / 1.46
    c2 = np.sqrt(1 - (np.sin(th) / n2) ** 2)
    r = (c1 - n2 * c2) / (c1 + n2 * c2) if name == "TE" else (n2 * c1 - c2) / (n2 * c1 + c2)
    ph = 2 * np.pi * 2.4 * c2 * L / lam
    t2 = abs((1 - r ** 2) * np.exp(1j * ph) / (1 - r ** 2 * np.exp(2j * ph))) ** 2
    check(f"oblique slab {name} vs Fresnel", abs(t1 - t2) < 1e-6, f"{t1:.6f} vs {t2:.6f}")
o = pb.simulate_pillars([600., 616.], twist_deg=0.0, theta_deg=8.0, phi_deg=0.0)
check("0 deg twist, tilt in mirror plane: CD ~ 0", np.abs(o.cd0).max() < 1e-2, f"{np.abs(o.cd0).max():.1e}")
o = pb.simulate_pillars([600., 616.], theta_deg=8.0, phi_deg=33.0)
check("oblique energy residual", o.residual.max() < 1e-3, f"{o.residual.max():.1e}")
o1 = pb.simulate_pillars([616.], theta_deg=0.01)
o0 = pb.simulate_pillars([616.])
check("theta -> 0 continuous", np.abs(o1.cd0 - o0.cd0).max() < 1e-3, f"{np.abs(o1.cd0 - o0.cd0).max():.1e}")
ws = np.array([600., 608., 616., 624.])
p = pb.simulate_pillars(ws)
m = pb.simulate_pillars(ws, twist_deg=-TW)
q = pb.simulate_pillars(ws, twist_deg=45.0)
z = pb.simulate_pillars(ws, twist_deg=0.0)
c = pb.simulate_pillars(ws, device="cpu")
check("energy residual", p.residual.max() < 1e-3, f"{p.residual.max():.1e}")
check("mirror twist flips sign", np.abs(p.cd0 + m.cd0).max() < 1e-3, f"{np.abs(p.cd0 + m.cd0).max():.1e}  cd0 {p.cd0.round(3)}")
check("45 deg CD ~ 0", np.abs(q.cd0).max() < 1e-2, f"{np.abs(q.cd0).max():.1e}")
check("0 deg CD ~ 0", np.abs(z.cd0).max() < 1e-2, f"{np.abs(z.cd0).max():.1e}  residual {z.residual.max():.1e}")
check("complex64 vs complex128", np.abs(p.cd0 - c.cd0).max() < 1e-3, f"{np.abs(p.cd0 - c.cd0).max():.1e}")
fine = pb.simulate_pillars(ws, n_cap=8, n_body=4)
print("slices 6 vs 12: max CD0 diff", np.abs(p.cd0 - fine.cd0).max())
for N in (1, 2, 3):
    t0 = time.time()
    r = pb.simulate_pillars([605., 606.], N=N)
    print(f"N={N}: {(time.time() - t0) / 2:.2f} s/wavelength  cd0 {r.cd0.round(3)}  residual {r.residual.max():.1e}", flush=True)
raise SystemExit(bad)
