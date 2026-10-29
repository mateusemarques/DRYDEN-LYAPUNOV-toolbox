"""Open-loop disturbance susceptibility of a generic longitudinal model (MLST).

States x = [u (m/s), alpha (rad), q (rad/s), theta (rad)], gust inputs
d = [u_g, w_g] (m/s), outputs y = [q, theta] (pilot-relevant).
All derivatives are ILLUSTRATIVE, not any specific aircraft.

Run:  python examples/mlst_example.py   (from the python/ folder)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drydenlyap import (modal_table, gramians, hankel_sv, h2_norm, hinf_norm,
                        sigma_max, dryden_params, augment, lyapunov_rms)

V, g = 150.0, 9.81
Xu, Xa = -0.015, 4.0
Zu, Za = -0.0009, -0.8
Mu, Ma, Mq = 0.0, -2.0, -0.9

A = np.array([[Xu, Xa, 0.0, -g],
              [Zu, Za, 1.0, 0.0],
              [Mu, Ma, Mq, 0.0],
              [0.0, 0.0, 1.0, 0.0]])
# aerodynamic velocities: u_air = u + u_g, alpha_air = alpha + w_g / V
B = np.column_stack([A[:, 0], A[:, 1] / V])
B[3, :] = 0.0
C = np.array([[0, 0, 1, 0], [0, 0, 0, 1.0]])
names = ["u", "alpha", "q", "theta"]

rows, eps = modal_table(A, B, C, state_names=names)
Wc, Wo, _ = gramians(A, B, C)
h2, _ = h2_norm(A, B, C)
hinf_all, w_all, _ = hinf_norm(A, B, C, band=(1e-3, 1e2))
hinf_pil, w_pil, _ = hinf_norm(A, B, C, band=(0.5, 10.0))

print(f"shift eps = {eps:g}")
print(f"{'mode':>6s} {'wn':>8s} {'zeta':>7s} {'Wc^':>8s} {'Wo^':>8s} {'index':>9s}  dominant")
for r in rows:
    print(f"{('osc' if r['oscillatory'] else 'real'):>6s} {r['wn']:8.4f} {r['zeta']:7.4f} "
          f"{r['wc_norm']:8.2e} {r['wo_norm']:8.2e} {r['index']:9.3e}  {r['dominant_state']}")
print("Hankel singular values:", np.round(hankel_sv(Wc, Wo), 4))
print(f"H2 = {h2:.4f}")
print(f"Hinf (all)          = {hinf_all:.3f} at {w_all:.4f} rad/s")
print(f"Hinf (0.5-10 rad/s) = {hinf_pil:.3f} at {w_pil:.3f} rad/s")

# Dryden-weighted view: RMS of q and theta in moderate turbulence at 300 m
p = dryden_params(V, 300.0, "moderate")
Bg = np.column_stack([B[:, 0], np.zeros(4), B[:, 1]])     # [u_g, v_g, w_g]
Aa, Ba, Ca = augment(A, Bg, C, p)
rms = lyapunov_rms(Aa, Ba, Ca) * 180 / np.pi
print(f"Dryden moderate, h = 300 m: RMS q = {rms[0]:.3f} deg/s, RMS theta = {rms[1]:.3f} deg")

w = np.logspace(-3, 2, 3000)
s = sigma_max(A - eps * np.eye(4), B, C, w)
fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.loglog(w, s)
ax.axvspan(0.5, 10, alpha=0.15, label="pilot band 0.5-10 rad/s")
ax.plot([w_all], [hinf_all], "o", label=f"global peak {hinf_all:.3f} @ {w_all:.3f} rad/s")
ax.plot([w_pil], [hinf_pil], "s", label=f"band peak {hinf_pil:.4f} @ {w_pil:.2f} rad/s")
ax.set_xlabel("frequency [rad/s]"); ax.set_ylabel("max singular value, gust -> [q, theta]")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "mlst_sigma.png")
fig.savefig(out, dpi=150)
