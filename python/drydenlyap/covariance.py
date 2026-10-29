"""Steady-state RMS by Lyapunov, Monte Carlo cross-check and band-limited variance."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm

from .dryden import dryden_ss, NOISE_INTENSITY, DrydenParams


def augment(A, Bg, C, p: DrydenParams, Dg=None):
    """Append the Dryden filters to a linear plant  x' = A x + Bg g,  y = C x + Dg g.

    Bg (n x 3) and Dg (ny x 3) map gust velocities [u_g, v_g, w_g] (m/s).
    Use zero columns for components the plant does not take.
    Returns (Aa, Ba, Ca) of the augmented system driven by 3 white noises.
    """
    A, Bg, C = np.atleast_2d(A), np.atleast_2d(Bg), np.atleast_2d(C)
    Af, Bf, Cf, _ = dryden_ss(p)
    n, m = A.shape[0], Af.shape[0]
    Dg = np.zeros((C.shape[0], 3)) if Dg is None else np.atleast_2d(Dg)
    Aa = np.block([[A, Bg @ Cf], [np.zeros((m, n)), Af]])
    Ba = np.vstack([np.zeros((n, 3)), Bf])
    Ca = np.hstack([C, Dg @ Cf])
    return Aa, Ba, Ca


def lyapunov_cov(Aa, Ba, W=NOISE_INTENSITY):
    """Stationary state covariance P:  Aa P + P Aa' + Ba W Ba' = 0."""
    if np.max(np.linalg.eigvals(Aa).real) >= 0:
        raise ValueError("augmented system is not asymptotically stable")
    Wm = W * np.eye(Ba.shape[1]) if np.isscalar(W) else W
    return solve_continuous_lyapunov(Aa, -Ba @ Wm @ Ba.T)


def lyapunov_rms(Aa, Ba, Ca, W=NOISE_INTENSITY):
    """Steady-state RMS of each output, no simulation needed."""
    P = lyapunov_cov(Aa, Ba, W)
    return np.sqrt(np.diag(Ca @ P @ Ca.T))


def discretize(Aa, Ba, dt, W=NOISE_INTENSITY):
    """Exact discretization (Van Loan): x[k+1] = Phi x[k] + w[k], w ~ N(0, Qd)."""
    Wm = W * np.eye(Ba.shape[1]) if np.isscalar(W) else W
    n = Aa.shape[0]
    M = np.block([[-Aa, Ba @ Wm @ Ba.T], [np.zeros((n, n)), Aa.T]]) * dt
    E = expm(M)
    Phi = E[n:, n:].T
    Qd = Phi @ E[:n, n:]
    return Phi, 0.5 * (Qd + Qd.T)


def monte_carlo_rms(Aa, Ba, Ca, dt=0.01, t_final=600.0, n_runs=20,
                    t_skip=None, seed=0, W=NOISE_INTENSITY):
    """Time-domain RMS of each output over n_runs seeds (mean, std).

    Starts each run from the stationary covariance, so no warm-up is
    strictly needed; t_skip discards an initial window anyway if given.
    """
    rng = np.random.default_rng(seed)
    Phi, Qd = discretize(Aa, Ba, dt, W)
    Lq = np.linalg.cholesky(Qd + 1e-30 * np.eye(len(Qd)))
    P0 = lyapunov_cov(Aa, Ba, W)  # exact sampling keeps the continuous stationary covariance
    L0 = np.linalg.cholesky(P0 + 1e-30 * np.eye(len(P0)))
    N = int(round(t_final / dt))
    k0 = 0 if t_skip is None else int(round(t_skip / dt))
    X = L0 @ rng.standard_normal((len(P0), n_runs))     # all runs at once
    acc = np.zeros((Ca.shape[0], n_runs))
    for k in range(N):
        if k >= k0:
            Y = Ca @ X
            acc += Y * Y
        X = Phi @ X + Lq @ rng.standard_normal(X.shape)
    out = np.sqrt(acc / (N - k0)).T
    return out.mean(axis=0), out.std(axis=0, ddof=1) if n_runs > 1 else np.zeros(Ca.shape[0])


def band_variance(Aa, Ba, Ca, w_lo, w_hi, n_pts=4000, W=NOISE_INTENSITY):
    """Fraction of each output's variance inside [w_lo, w_hi] rad/s.

    Uses the one-sided PSD S(w) = |G(jw)|^2 W / pi, which integrates to the
    Lyapunov variance over w in [0, inf).
    """
    Wm = W * np.eye(Ba.shape[1]) if np.isscalar(W) else W
    w = np.logspace(-5, 4, n_pts)
    I = np.eye(Aa.shape[0])
    S = np.empty((len(w), Ca.shape[0]))
    for i, wi in enumerate(w):
        G = Ca @ np.linalg.solve(1j * wi * I - Aa, Ba)
        S[i] = np.einsum("ij,jk,ik->i", G, Wm, G.conj()).real / np.pi
    tot = np.trapezoid(S, w, axis=0)
    m = (w >= w_lo) & (w <= w_hi)
    return np.trapezoid(S[m], w[m], axis=0) / tot, w, S
