import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drydenlyap import (dryden_params, dryden_ss, augment, lyapunov_rms,
                        lyapunov_cov, monte_carlo_rms, band_variance, NOISE_INTENSITY)


@pytest.mark.parametrize("h,sig_hi", [(50.0, None), (300.0, None), (450.0, 1.5), (3000.0, 2.0)])
@pytest.mark.parametrize("intensity", ["light", "moderate", "severe"])
def test_filter_variance_equals_sigma2(h, sig_hi, intensity):
    p = dryden_params(100.0, h, intensity, sigma_high=sig_hi)
    A, B, C, _ = dryden_ss(p)
    P = lyapunov_cov(A, B, NOISE_INTENSITY)
    np.testing.assert_allclose(np.sqrt(np.diag(C @ P @ C.T)), p.sigma, rtol=1e-9)


def test_low_altitude_formulas():
    p = dryden_params(80.0, 1000 * 0.3048, "moderate")
    W20 = 30 * 0.514444
    assert p.sigma[2] == pytest.approx(0.1 * W20)
    assert p.L[2] == pytest.approx(1000 * 0.3048)
    assert p.L[0] == pytest.approx(1000 / (0.177 + 0.000823 * 1000) ** 1.2 * 0.3048)


def test_high_altitude_requires_sigma():
    with pytest.raises(ValueError):
        dryden_params(200.0, 5000.0)
    p = dryden_params(200.0, 5000.0, sigma_high=1.5)
    assert p.L == pytest.approx((1750 * 0.3048,) * 3) and p.sigma == (1.5,) * 3


def _plant():
    V = 80.0
    A = np.array([[-1.0, 1.0], [-3.0, -1.2]])
    Bg = np.array([[0, 0, -1.0 / V], [0, 0, -3.0 / V]])
    return A, Bg, np.eye(2), dryden_params(V, 300.0, "moderate")


def test_monte_carlo_matches_lyapunov():
    A, Bg, C, p = _plant()
    Aa, Ba, Ca = augment(A, Bg, C, p)
    rl = lyapunov_rms(Aa, Ba, Ca)
    mm, _ = monte_carlo_rms(Aa, Ba, Ca, dt=0.01, t_final=400, n_runs=16, seed=3)
    np.testing.assert_allclose(mm, rl, rtol=0.05)


def test_band_fraction_full_band_is_one():
    A, Bg, C, p = _plant()
    Aa, Ba, Ca = augment(A, Bg, C, p)
    frac, _, _ = band_variance(Aa, Ba, Ca, 1e-5, 1e4)
    np.testing.assert_allclose(frac, 1.0, rtol=1e-3)


def test_unstable_plant_rejected():
    A, Bg, C, p = _plant()
    Aa, Ba, Ca = augment(-A, Bg, C, p)
    with pytest.raises(ValueError):
        lyapunov_rms(Aa, Ba, Ca)
