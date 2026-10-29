"""Open-loop disturbance susceptibility with Multivariable Linear Systems Theory (MLST).

Tools for a linear model  x' = A x + B d,  y = C x  (d = disturbance inputs,
y = pilot-relevant outputs):

* gramians       controllability Wc and observability Wo (Lyapunov)
* modal_table    each mode's wn, zeta and its modal reachability from d and
                 visibility in y (diagonal of the Gramians in modal coordinates)
* hankel_sv      Hankel singular values (input-output energy ranking)
* h2_norm        aggregate (RMS) amplification for unit-intensity white noise
* hinf_norm      worst-case gain and its frequency, optionally inside a band

Models with marginal modes (integrators such as heading or position, or a
neutral spiral) have no finite Gramians; pass `shift` (or let it default) to
evaluate the slightly shifted system A - eps I, and report eps with results.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eig, svdvals, solve
from scipy.optimize import minimize_scalar


def stability_shift(A, margin=5e-3, tol=1e-6):
    """eps such that A - eps I is Hurwitz (0 when A already is)."""
    mr = np.max(np.linalg.eigvals(A).real)
    return 0.0 if mr < -tol else float(mr + margin)


def gramians(A, B, C, shift=None):
    """Controllability and observability Gramians of (A - eps I, B, C)."""
    A, B, C = map(np.atleast_2d, (A, B, C))
    eps = stability_shift(A) if shift is None else shift
    As = A - eps * np.eye(len(A))
    Wc = solve_continuous_lyapunov(As, -B @ B.T)
    Wo = solve_continuous_lyapunov(As.T, -C.T @ C)
    return 0.5 * (Wc + Wc.T), 0.5 * (Wo + Wo.T), eps


def hankel_sv(Wc, Wo):
    """Hankel singular values, descending."""
    ev = np.linalg.eigvals(Wc @ Wo)
    return np.sort(np.sqrt(np.abs(ev.real)))[::-1]


def h2_norm(A, B, C, shift=None):
    """H2 norm sqrt(trace(C Wc C')): RMS of y for unit-intensity white noise on every input."""
    Wc, _, eps = gramians(A, B, C, shift)
    return float(np.sqrt(np.trace(C @ Wc @ C.T))), eps


def sigma_max(A, B, C, w, D=None):
    """Largest singular value of G(jw) = C (jwI - A)^-1 B + D over the frequencies w."""
    n = len(A)
    D = np.zeros((C.shape[0], B.shape[1])) if D is None else D
    return np.array([svdvals(C @ solve(1j * x * np.eye(n) - A, B) + D)[0] for x in np.atleast_1d(w)])


def hinf_norm(A, B, C, band=(1e-3, 1e2), n_grid=2000, shift=None, D=None):
    """Peak of sigma_max inside band (rad/s) and the frequency where it occurs.

    Grid search on a log scale, then a bounded local refinement.
    """
    eps = stability_shift(A) if shift is None else shift
    As = A - eps * np.eye(len(A))
    w = np.logspace(np.log10(band[0]), np.log10(band[1]), n_grid)
    s = sigma_max(As, B, C, w, D)
    k = int(np.argmax(s))
    lo, hi = w[max(k - 1, 0)], w[min(k + 1, len(w) - 1)]
    if hi > lo:
        r = minimize_scalar(lambda x: -sigma_max(As, B, C, [x], D)[0],
                            bounds=(lo, hi), method="bounded")
        if -r.fun > s[k]:
            return float(-r.fun), float(r.x), eps
    return float(s[k]), float(w[k]), eps


def modal_table(A, B, C, state_names=None, shift=None, tol=1e-8):
    """One row per mode (complex pairs merged), sorted by natural frequency.

    Each row: eigenvalue, wn [rad/s], zeta, wc (modal reachability from B),
    wo (modal visibility in C), their values normalised by the largest mode,
    modal Hankel-like index sqrt(wc*wo), and the dominant state (largest
    participation) when state_names is given.
    """
    A, B, C = map(np.atleast_2d, (A, B, C))
    Wc, Wo, eps = gramians(A, B, C, shift)
    lam, Phi = eig(A)
    Pinv = np.linalg.inv(Phi)
    wc_k = np.real(np.diag(Pinv @ Wc @ Pinv.conj().T))
    wo_k = np.real(np.diag(Phi.conj().T @ Wo @ Phi))
    part = np.abs(Phi * Pinv.T)                      # participation factors
    used, rows = set(), []
    for k in range(len(lam)):
        if k in used or lam[k].imag < -tol:
            continue
        group = [k]
        if abs(lam[k].imag) > tol:
            j = [j for j in range(len(lam)) if j != k and j not in used
                 and abs(lam[j] - np.conj(lam[k])) < 1e-6 * max(1, abs(lam[k]))]
            group += j[:1]
        used.update(group)
        wn = abs(lam[k])
        row = dict(eig=complex(lam[k]), wn=float(wn),
                   zeta=float(-lam[k].real / wn) if wn > tol else float("nan"),
                   oscillatory=abs(lam[k].imag) > tol,
                   wc=float(sum(wc_k[g] for g in group)),
                   wo=float(sum(wo_k[g] for g in group)))
        if state_names is not None:
            row["dominant_state"] = state_names[int(np.argmax(part[:, k]))]
        rows.append(row)
    wc_max = max(abs(r["wc"]) for r in rows) or 1.0
    wo_max = max(abs(r["wo"]) for r in rows) or 1.0
    for r in rows:
        r["wc_norm"] = r["wc"] / wc_max
        r["wo_norm"] = r["wo"] / wo_max
        r["index"] = float(np.sqrt(abs(r["wc"] * r["wo"])))
    rows.sort(key=lambda r: r["wn"])
    return rows, eps
