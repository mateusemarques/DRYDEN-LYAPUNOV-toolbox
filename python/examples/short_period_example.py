"""Generic short-period aircraft in vertical Dryden turbulence.

Plant values are ILLUSTRATIVE (not any specific aircraft):
    alpha' = Za (alpha + w_g/V) + q
    q'     = Ma (alpha + w_g/V) + Mq q
    dnz    = -(V/g) Za (alpha + w_g/V)           [g]

Run:  python examples/short_period_example.py   (from the python/ folder)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drydenlyap import (dryden_params, augment, lyapunov_rms,
                        monte_carlo_rms, band_variance)

V, h, g = 80.0, 300.0, 9.81          # m/s, m (about 984 ft, low-altitude band)
Za, Ma, Mq = -1.0, -3.0, -1.2        # 1/s, 1/s^2, 1/s  (illustrative)

A = np.array([[Za, 1.0], [Ma, Mq]])
Bg = np.array([[0, 0, Za / V], [0, 0, Ma / V]])      # columns: u_g, v_g, w_g
C = np.array([[1, 0], [0, 1], [-(V / g) * Za, 0]])   # alpha, q, nz
Dg = np.array([[0, 0, 0], [0, 0, 0], [0, 0, -(V / g) * Za / V]])
names = ["alpha [deg]", "q [deg/s]", "nz [g]"]
scale = np.array([180 / np.pi, 180 / np.pi, 1.0])

here = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(here, "..", "..", "docs")
os.makedirs(fig_dir, exist_ok=True)

rows = []
for i, intensity in enumerate(["light", "moderate", "severe"]):
    p = dryden_params(V, h, intensity)
    Aa, Ba, Ca = augment(A, Bg, C, p, Dg)
    rms_lyap = lyapunov_rms(Aa, Ba, Ca) * scale
    mc_mean, mc_std = monte_carlo_rms(Aa, Ba, Ca, dt=0.01, t_final=600, n_runs=20, seed=1 + i)
    rows.append((intensity, p, rms_lyap, mc_mean * scale, mc_std * scale))

print(f"V = {V} m/s, h = {h} m")
for intensity, p, rl, mm, ms in rows:
    print(f"\n{intensity}: sigma_w = {p.sigma[2]:.3f} m/s, L_w = {p.L[2]:.1f} m")
    for n, a, b, c in zip(names, rl, mm, ms):
        print(f"  {n:12s} Lyapunov {a:8.4f} | Monte Carlo {b:8.4f} +- {c:.4f} "
              f"| diff {100*(b-a)/a:+.1f}%")

# Figure 1: Lyapunov vs Monte Carlo
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
x = np.arange(3)
for j in range(3):
    ax[j].bar(x - 0.18, [r[2][j] for r in rows], 0.36, label="Lyapunov")
    ax[j].bar(x + 0.18, [r[3][j] for r in rows], 0.36,
              yerr=[r[4][j] for r in rows], capsize=3, label="Monte Carlo (20 runs)")
    ax[j].set_xticks(x, [r[0] for r in rows]); ax[j].set_title("RMS " + names[j])
    ax[j].grid(alpha=0.3, axis="y")
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "lyapunov_vs_mc.png"), dpi=150)

# Figure 2: one-sided PSD of q (moderate) and fraction inside the pilot band
p = rows[1][1]
Aa, Ba, Ca = augment(A, Bg, C, p, Dg)
frac, w, S = band_variance(Aa, Ba, Ca, 1.0, 10.0)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.loglog(w, S[:, 1] * scale[1] ** 2)
ax.axvspan(1, 10, alpha=0.15, label=f"1-10 rad/s: {100*frac[1]:.0f}% of var(q)")
ax.set_xlim(1e-2, 1e2); ax.set_ylim(bottom=1e-6)
ax.set_xlabel("frequency [rad/s]"); ax.set_ylabel("one-sided PSD of q [(deg/s)^2/(rad/s)]")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "psd_q_band.png"), dpi=150)
print("\nfigures saved in docs/")
