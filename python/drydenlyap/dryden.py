"""Dryden turbulence model (MIL-F-8785C / MIL-HDBK-1797), frozen V and h.

Shaping filters (driven by unit-intensity-pi white noise, see README):

    H_u(s) = sigma_u * sqrt(2 L_u / (pi V)) / (1 + (L_u/V) s)
    H_v(s) = sigma_v * sqrt(L_v / (pi V)) * (1 + sqrt(3) (L_v/V) s) / (1 + (L_v/V) s)^2
    H_w(s) = same form as H_v with L_w, sigma_w

All quantities in SI units (m, m/s). Altitude bands follow MIL-HDBK-1797:
low altitude (h <= 1000 ft) from the 20 ft wind speed W20, medium/high
altitude (h >= 2000 ft) with L = 1750 ft and a user-supplied sigma
(read it from the handbook's intensity-vs-altitude chart for the
exceedance probability you need), linear interpolation in between.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FT = 0.3048
KT = 0.514444
W20_KT = {"light": 15.0, "moderate": 30.0, "severe": 45.0}


@dataclass
class DrydenParams:
    V: float                 # true airspeed [m/s]
    h: float                 # altitude [m]
    L: tuple                 # (Lu, Lv, Lw) [m]
    sigma: tuple             # (su, sv, sw) [m/s]


def _low_altitude(h_ft: float, W20: float):
    den = 0.177 + 0.000823 * h_ft
    Lw = h_ft * FT
    Lu = h_ft / den ** 1.2 * FT
    sw = 0.1 * W20
    su = sw / den ** 0.4
    return (Lu, Lu, Lw), (su, su, sw)


def dryden_params(V: float, h: float, intensity: str = "moderate",
                  sigma_high: float | None = None) -> DrydenParams:
    """Scale lengths and intensities for airspeed V [m/s] and altitude h [m].

    intensity  : 'light' | 'moderate' | 'severe' (sets W20 = 15/30/45 kt)
    sigma_high : sigma [m/s] for h >= 2000 ft. Required when h > 1000 ft.
    """
    if intensity not in W20_KT:
        raise ValueError("intensity must be 'light', 'moderate' or 'severe'")
    if V <= 0:
        raise ValueError("V must be positive")
    W20 = W20_KT[intensity] * KT
    h_ft = max(h / FT, 10.0)
    if h_ft <= 1000:
        L, s = _low_altitude(h_ft, W20)
    else:
        if sigma_high is None:
            raise ValueError("sigma_high is required above 1000 ft "
                             "(read it from MIL-HDBK-1797, intensity vs altitude)")
        L_hi = (1750 * FT,) * 3
        s_hi = (float(sigma_high),) * 3
        if h_ft >= 2000:
            L, s = L_hi, s_hi
        else:
            L1, s1 = _low_altitude(1000.0, W20)
            f = (h_ft - 1000.0) / 1000.0
            L = tuple(a + f * (b - a) for a, b in zip(L1, L_hi))
            s = tuple(a + f * (b - a) for a, b in zip(s1, s_hi))
    return DrydenParams(V=V, h=h, L=tuple(L), sigma=tuple(s))


def _ss_first_order(L, sigma, V):
    tau = L / V
    k = sigma * np.sqrt(2 * L / (np.pi * V))
    return (np.array([[-1 / tau]]), np.array([[k / tau]]),
            np.array([[1.0]]), np.zeros((1, 1)))


def _ss_second_order(L, sigma, V):
    tau = L / V
    k = sigma * np.sqrt(L / (np.pi * V))
    A = np.array([[0.0, 1.0], [-1 / tau ** 2, -2 / tau]])
    B = np.array([[0.0], [1.0]])
    C = k * np.array([[1 / tau ** 2, np.sqrt(3) / tau]])
    return A, B, C, np.zeros((1, 1))


def dryden_ss(p: DrydenParams):
    """Block-diagonal state-space (A, B, C, D) of the three Dryden filters.

    Inputs : 3 white noises (u, v, w channels), intensity pi each.
    Outputs: gust velocities [u_g, v_g, w_g] in m/s. 5 states.
    """
    blocks = [_ss_first_order(p.L[0], p.sigma[0], p.V),
              _ss_second_order(p.L[1], p.sigma[1], p.V),
              _ss_second_order(p.L[2], p.sigma[2], p.V)]
    from scipy.linalg import block_diag
    A = block_diag(*[b[0] for b in blocks])
    B = block_diag(*[b[1] for b in blocks])
    C = block_diag(*[b[2] for b in blocks])
    D = np.zeros((3, 3))
    return A, B, C, D


NOISE_INTENSITY = np.pi
"""Continuous white-noise intensity that makes each filter output have variance sigma^2."""
